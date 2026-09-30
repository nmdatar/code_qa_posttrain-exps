"""Bounded preservation download via existing controller REST credentials only.

No trainer creation, sampling, load/save, optimizer calls, TTL updates, ledger
writes or remote filesystem writes. Signed URLs stay in process memory.
"""
import argparse
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import struct
import tarfile
import time
import urllib.request
import uuid

MAX_BYTES = 8 * 1024**3


def strict_json(raw):
    def unique(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=unique)


def verify_tar(path, expected_rank):
    with tarfile.open(path, mode='r:') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError('Unsafe archive member')
        required = {'adapter_config.json', 'adapter_model.safetensors', 'checkpoint_complete'}
        if set(names) != required:
            raise ValueError('Unexpected checkpoint archive members')
        if archive.getmember('adapter_config.json').size > 1024**2:
            raise ValueError('Oversized adapter configuration')
        config = strict_json(archive.extractfile('adapter_config.json').read())
        if config.get('peft_type') != 'LORA' or config.get('r') != expected_rank:
            raise ValueError('Unexpected adapter type or rank')
        model = archive.getmember('adapter_model.safetensors')
        stream = archive.extractfile(model)
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Missing safetensors header')
        length = struct.unpack('<Q', prefix)[0]
        if length <= 0 or length > 16 * 1024**2 or 8 + length >= model.size:
            raise ValueError('Invalid safetensors header length')
        header = strict_json(stream.read(length))
        byte_sizes = {'BOOL': 1, 'U8': 1, 'I8': 1, 'I16': 2, 'U16': 2,
                      'F16': 2, 'BF16': 2, 'I32': 4, 'U32': 4, 'F32': 4,
                      'I64': 8, 'U64': 8, 'F64': 8}
        ranges = []
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            shape, offsets, dtype = tensor['shape'], tensor['data_offsets'], tensor['dtype']
            if (dtype not in byte_sizes or not isinstance(shape, list) or
                    any(type(d) is not int or d < 0 for d in shape) or
                    not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(n) is not int for n in offsets)):
                raise ValueError('Invalid tensor metadata')
            start, end = offsets
            if start < 0 or end < start or end - start != math.prod(shape) * byte_sizes[dtype]:
                raise ValueError('Invalid tensor data range')
            ranges.append((start, end))
        if not ranges:
            raise ValueError('No adapter tensors')
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError('Overlapping or unaccounted tensor data')
            cursor = end
        if cursor != model.size - 8 - length:
            raise ValueError('Tensor data does not cover archive member')
        return {'members': names, 'adapter_rank': config['r'], 'tensor_count': len(ranges),
                'safetensors_bytes': model.size, 'structure_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sandbox-id', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, choices=[42, 44], required=True)
    args = parser.parse_args()
    logging.getLogger('modal').setLevel(logging.ERROR)
    import modal
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path('artifacts/fixed-final-sampler-exports') / (timestamp + '-' + uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'directory': str(root.resolve()), 'exports': []}
    sandbox = modal.Sandbox.from_id(args.sandbox_id)
    try:
        for seed in args.seeds:
            receipt = {'seed': seed, 'status': 'pending', 'trainer_created': False,
                       'optimizer_calls': 0, 'ttl_mutations': 0, 'remote_filesystem_writes': 0}
            target = root / f'seed{seed}'
            target.mkdir()
            try:
                report = strict_json(Path(f'reports/expanded-studies/direct-seed{seed}-results.json').read_bytes())
                manifest_path = Path(report['checkpoint']['path'])
                raw = manifest_path.read_bytes()
                if report['input_sha256'].get(str(manifest_path)) != hashlib.sha256(raw).hexdigest():
                    raise ValueError('Final checkpoint manifest hash differs from results audit')
                manifest = strict_json(raw)
                if (manifest['id'] != report['evaluations']['final']['checkpoint_id'] or
                        manifest['state']['optimizer_step'] != 32):
                    raise ValueError('Not the fixed final checkpoint')
                sampler = manifest['artifacts']['sampler']
                if sampler.split('/')[-1] != manifest['id'] or '/sampler_weights/' not in sampler:
                    raise ValueError('Sampler path does not match final checkpoint')
                receipt.update(checkpoint_id=manifest['id'], optimizer_step=32, sampler_uri=sampler,
                               manifest_path=str(manifest_path), manifest_sha256=hashlib.sha256(raw).hexdigest())
                if sandbox.poll() is not None:
                    raise RuntimeError('Existing controller is terminal; no replacement launch permitted')
                # This stdout is captured privately and never forwarded to logs.
                remote = """import json,logging,sys
logging.disable(logging.CRITICAL)
try:
 import tinker
 service=tinker.ServiceClient(timeout=20,max_retries=0)
 result=service.create_rest_client().get_checkpoint_archive_url_from_tinker_path(sys.argv[1]).result(timeout=25)
 print(json.dumps({'url':result.url}))
except Exception as exc:
 print(json.dumps({'error_type':type(exc).__name__}))
"""
                process = sandbox.exec('python', '-B', '-c', remote, sampler, timeout=30)
                captured = process.stdout.read()
                process.wait()
                payload = strict_json(captured)
                if 'url' not in payload:
                    raise RuntimeError('Archive URL unavailable: ' + payload.get('error_type', 'unknown'))
                url = payload.pop('url')
                if not isinstance(url, str) or not url.startswith('https://'):
                    raise ValueError('Archive URL must use HTTPS')
                pending = target / 'sampler.pending'
                digest, size = hashlib.sha256(), 0
                deadline = time.monotonic() + 300
                with urllib.request.urlopen(url, timeout=60) as response, pending.open('xb') as output:
                    while chunk := response.read(1024**2):
                        if time.monotonic() > deadline:
                            raise TimeoutError('Download deadline exceeded')
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError('Archive exceeds 8 GiB bound')
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                del url, captured
                if not size:
                    raise ValueError('Empty checkpoint archive')
                validation = verify_tar(pending, manifest['identity']['rank'])
                with pending.open('rb') as stream:
                    rehash = hashlib.file_digest(stream, 'sha256').hexdigest()
                if rehash != digest.hexdigest():
                    raise ValueError('Archive readback hash mismatch')
                archive = target / 'sampler.tar'
                os.replace(pending, archive)
                receipt.update(status='verified', archive_path=str(archive.resolve()), bytes=size,
                               sha256=rehash, validation=validation, optimizer_state_exported=False,
                               tinker_archive_reimport_verified=False)
            except Exception as exc:
                # Never emit exception text: HTTP/provider errors may contain URLs.
                receipt.update(status='failed', error_type=type(exc).__name__,
                               action='No replay or mutation attempted; inspect categorical failure separately')
            (target / 'export-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            summary['exports'].append(receipt)
            print(json.dumps({k:receipt[k] for k in ('seed','status','checkpoint_id','bytes','sha256','error_type') if k in receipt}), flush=True)
    finally:
        sandbox.detach()
        summary['status'] = 'verified' if len(summary['exports']) == len(args.seeds) and all(x['status']=='verified' for x in summary['exports']) else 'incomplete'
        (root / 'export-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'status':summary['status'], 'summary_path':str((root/'export-summary.json').resolve())}), flush=True)


if __name__ == '__main__':
    main()
