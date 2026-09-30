"""Build and verify a versioned development batch; never auto-approve gold.

Run with python -m dataset_builder.batch --config ... --phase prepare|build|verify|package.
Investigation attempts are driven through dataset_builder.investigate separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .build import (prepare, build_environment_bundle, read_jsonl, write_json,
                    write_jsonl)
from .contracts import canonical_hash, validate_bundle
from .verify import verify_bundle
from .isolation import check_isolation
from .attest import attest


def load(path):
    return json.loads(Path(path).read_text())


def _check_hashes(root, manifest):
    for name, expected in manifest['artifacts'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Manifest path escapes bundle')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Bundle artifact changed: ' + name)


def audit_batch(config, root):
    """Bind records, immutable builds, observed verification and family splits."""
    public, private, environments, bundles = [], [], [], []
    seen_families = set()
    for family in config['families']:
        if family['split'] != 'development':
            raise ValueError('This release path is development-only')
        if family['family_id'] in seen_families:
            raise ValueError('Duplicate batch family')
        seen_families.add(family['family_id'])
        bundle = root / family['bundle']
        manifest = load(bundle / 'manifest.json')
        _check_hashes(bundle, manifest)
        repository = {'url': family['repository'], 'commit': family['commit'],
                      'family_id': family['family_id']}
        if manifest['repository'] != repository or manifest['split'] != family['split']:
            raise ValueError('Batch family assignment differs from prepared bundle')
        env = load(bundle / 'public/environment.json')
        built = load(bundle / 'environment-build/result.json')
        verified = load(bundle / 'private/verification-report.json')
        isolation = load(bundle / 'private/isolation-report.json')
        if built['status'] != 'ready' or built['commit'] != family['commit']:
            raise ValueError('Unready or wrong-commit environment')
        if built.get('source_environment_sha256') != canonical_hash(env):
            raise ValueError('Image recipe binding mismatch')
        if verified.get('bundle_hash') != canonical_hash(manifest) or verified.get('environment_hash') != canonical_hash(built):
            raise ValueError('Stale assertion report')
        assertions = verified.get('assertions', [])
        if verified.get('status') != 'passed' or not assertions or any(a['status'] != 'passed' for a in assertions):
            raise ValueError('Assertions must pass before packaging runnable tasks')
        if not isolation.get('passed') or isolation.get('environment_hash') != canonical_hash(built):
            raise ValueError('Isolation checks missing or not bound to this image')
        attestation=load(bundle / 'private/image-attestation.json')
        if attestation.get('status')!='passed' or attestation.get('environment_hash')!=canonical_hash(built):
            raise ValueError('Image source attestation missing or stale')
        p = read_jsonl(bundle / 'public/tasks.jsonl')
        q = read_jsonl(bundle / 'private/tasks.jsonl')
        public.extend(p); private.extend(q); environments.append(env)
        bundles.append((bundle, built, verified, isolation))
    validate_bundle(public, private, environments)
    return public, private, environments, bundles


def package(config, root, output=None):
    root = Path(root)
    public, private, environments, bundles = audit_batch(config, root)
    output = Path(output) if output else root / config['release']
    if output.exists() and any(output.iterdir()):
        raise ValueError('Release directory not empty; use a new version')
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output/'evaluation/development/tasks.jsonl', public)
    write_jsonl(output/'private/references.jsonl', private)
    image_refs=[]
    for env, (bundle, built, verified, isolation) in zip(environments, bundles):
        image_refs.append({'environment_id':env['id'], 'repository':env['repository'],
                           'backend':built['backend'], 'image_digest':built['image_digest'],
                           'recipe_sha256':built['recipe_sha256'],
                           'snapshot_sha256':built['snapshot_sha256'],
                           'tool_version':built['tool_version'],
                           'capability':built['capability']})
        prefix=output/'private'/env['id']
        write_json(prefix/'verification.json',verified)
        write_json(prefix/'isolation.json',isolation)
        write_json(prefix/'build.json',built)
        shutil.copyfile(bundle/'private/source-spec.json', prefix/'source-spec.json')
        shutil.copyfile(bundle/'private/assertions.jsonl', prefix/'assertions.jsonl')
        if (bundle/'private/image-attestation.json').exists():
            attestation=load(bundle/'private/image-attestation.json')
            if attestation.get('environment_hash')!=canonical_hash(built) or attestation.get('status')!='passed':
                raise ValueError('Invalid image attestation')
            write_json(prefix/'image-attestation.json',attestation)
    write_jsonl(output/'evaluation/development/environments.jsonl',image_refs)
    for name in ['sft/train.jsonl','rl/train.jsonl','preferences/train.jsonl','evaluation/final_test/tasks.jsonl']:
        write_jsonl(output/name,[])
    write_json(output/'split-manifest.json',{'id':config['id'], 'families':config['families']})
    card=(
        '# Repository Q&A development release\n\n'
        f'{len(public)} newly authored task records, pinned repository environments, and observed private behavior checks. '
        'This release is runnable development data, not human-approved gold or a training release.\n\n'
        'SFT, RL, preference and final-test outputs are intentionally empty: every selected family is development-only. '
        'Keep private/ outside agent inputs. The solver receives tasks.jsonl and resolved environment references only.\n\n'
        'Fresh isolated execution per tool operation; mutable /tmp state does not persist between calls. '
        'Stateful experiments run as one bounded operation. Reuse immutable Modal image references for replay; '
        'a rebuild from a base tag can differ. Code licenses and provenance are preserved in private source specifications.\n\n'
        'Required before human-approved evaluation: reference review and adjudication. '
        'Full benchmark/semantic overlap checks remain incomplete; no training exports are allowed.\n'
    )
    (output/'DATASET_CARD.md').write_text(card)
    files={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(output.rglob('*')) if p.is_file()}
    manifest={'schema_version':'1.0', 'release_id':config['id'],
              'status':'runnable_development_preview', 'config_sha256':canonical_hash(config),
              'task_count':len(public), 'family_count':len(environments),
              'task_ids':[t['id'] for t in public], 'human_reviewed':False,
              'training_eligible':False, 'gold_status':'draft',
              'counts':{'development':len(public),'final_test':0,'sft':0,'rl':0,'preferences':0},
              'artifacts':files}
    write_json(output/'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','build','verify','package'],required=True)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();config=load(args.config);root=args.root.resolve()
    if args.phase=='package':
        print(json.dumps(package(config,root,args.output),indent=2));return
    results=[]
    for family in config['families']:
        bundle=root/family['bundle']
        try:
            if args.phase=='prepare':
                spec=load(root/family['spec'])
                if spec['split']!=family['split'] or spec['repository']['family_id']!=family['family_id'] or spec['repository']['commit']!=family['commit']:
                    raise ValueError('Spec differs from preassigned family')
                if (bundle/'manifest.json').exists():
                    manifest=load(bundle/'manifest.json')
                    if manifest['source_spec_sha256']!=canonical_hash(spec):
                        raise ValueError('Spec changed; use new bundle version')
                    _check_hashes(bundle,manifest)
                    result={'status':'reused_prepared_bundle'}
                else: result=prepare(root/family['spec'],bundle,root/'artifacts/repos')
            elif args.phase=='build':
                result=build_environment_bundle(bundle,config['backend'])
            else:
                result={'verification':verify_bundle(bundle), 'isolation':check_isolation(bundle), 'image_attestation':attest(bundle)}
                result['status']='passed' if result['verification']['status']=='passed' and result['isolation']['passed'] and result['image_attestation']['status']=='passed' else 'not_passed'
            results.append({'family':family['family_id'],'status':result.get('status'),'result':result})
        except Exception as exc:
            results.append({'family':family['family_id'],'status':'blocked','error':str(exc)})
    path=root/'artifacts'/config['id']/(args.phase+'.json')
    write_json(path,results)
    print(json.dumps({'report':str(path),'families':[{'family':r['family'],'status':r['status']} for r in results]},indent=2))
    if any(r['status'] in {'blocked','not_passed','quarantined'} for r in results):raise SystemExit(1)


if __name__=='__main__':main()
