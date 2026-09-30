"""Source-only Git-object environments, independent of host checkout conversions.

Regular tracked blobs are exposed exactly. Symlink targets and submodule commit
IDs are metadata, not dereferenced source. Repository code is never executed.
"""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import io

from .environment import _git, _git_blobs, _modal_module, ModalBackend, RunLimits
from .contracts import canonical_hash
from .build import write_json


def source_objects(root, commit):
    if _git(root,'rev-parse','HEAD').decode().strip()!=commit:raise ValueError('Wrong Git commit')
    entries=[];links={};submodules={}
    for item in _git(root,'ls-tree','-rz','--full-tree',commit).split(b'\0'):
        if not item:continue
        metadata,name=item.split(b'\t',1);mode,kind,oid=metadata.decode().split();name=name.decode('utf-8')
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe Git path')
        if mode=='160000' and kind=='commit':submodules[name]={'commit':oid,'available':False};continue
        if kind!='blob' or mode not in ['100644','100755','120000']:raise ValueError('Unsupported Git object')
        entries.append((name,mode,oid))
    blobs=_git_blobs(root,(oid for _,_,oid in entries));files={};modes={}
    for name,mode,oid in entries:
        if mode=='120000':links[name]={'target':blobs[oid].decode('utf-8'),'dereferenced':False};continue
        files[name]=blobs[oid];modes[name]=mode
    return files,modes,links,submodules


def inventory(root,commit):
    files,modes,links,submodules=source_objects(root,commit)
    return {'snapshot_files':{p:hashlib.sha256(v).hexdigest() for p,v in files.items()},
            'symbolic_links':links,'submodules':submodules,
            'git_tree':_git(root,'rev-parse',commit+'^{tree}').decode().strip()}


def read_git_file(root,commit,path):
    if path.startswith('/') or '..' in PurePosixPath(path).parts or '\\' in path:raise ValueError('Unsafe Git path')
    return _git(root,'show',commit+':'+path)


def build_source_environment(root,commit,recipe,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    files,modes,links,submodules=source_objects(root,commit)
    hashes={p:hashlib.sha256(v).hexdigest() for p,v in files.items()}
    if recipe.capability!='source_reading' or recipe.install_commands:raise ValueError('Git source projection is source-reading only')
    modal=_modal_module();backend=ModalBackend()
    with tempfile.TemporaryDirectory(prefix='qa-git-source-') as tmp:
        archive=Path(tmp)/'source.tar'
        with tarfile.open(archive,'w') as tar:
            for path,data in sorted(files.items()):
                entry=tarfile.TarInfo(path);entry.size=len(data);entry.mode=0o444;entry.mtime=0
                tar.addfile(entry,io.BytesIO(data))
        # Only regular entries from our validated Git object inventory are archived.
        extractor="import tarfile,pathlib; p=pathlib.Path('/workspace'); p.mkdir(exist_ok=True); t=tarfile.open('/opt/source.tar'); assert all(m.isfile() and not m.name.startswith('/') and '..' not in pathlib.PurePosixPath(m.name).parts for m in t.getmembers()); t.extractall(p,filter='data'); t.close(); pathlib.Path('/opt/source.tar').unlink()"
        import shlex
        image=modal.Image.from_registry(recipe.base_image).add_local_file(archive,'/opt/source.tar',copy=True)
        image=image.run_commands('python -I -c '+shlex.quote(extractor),'chmod -R a-w /workspace').workdir('/workspace').dockerfile_commands('USER 65534:65534','ENV PYTHONDONTWRITEBYTECODE=1')
        app=modal.App('dataset-git-source-build')
        with app.run():
            image.build(app);image_id=image.object_id
    readiness=backend.run(image_id,recipe.readiness_command,RunLimits(timeout_seconds=60))
    recipe_record={'base_image':recipe.base_image,'capability':'source_reading','source_inventory_mode':'git_objects_primary','install_commands':[],'readiness_command':recipe.readiness_command,'extractor':extractor}
    built={'schema_version':'1','environment_id':'env-'+image_id,'commit':commit,'capability':'source_reading','image_digest':image_id,'image_id':image_id,'backend':'modal',
        'source_inventory_mode':'git_objects_primary','source_scope':'All primary-repository regular Git blobs; symlinks and submodule pins are explicit metadata and are not dereferenced. No repository execution.',
        'snapshot_files':hashes,'symbolic_links':links,'submodules':submodules,'git_tree':_git(root,'rev-parse',commit+'^{tree}').decode().strip(),
        'materialized_symlinks':{},'snapshot_sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        'recipe_sha256':canonical_hash(recipe_record),'tool_version':'dataset-git-source-v1','readiness':readiness,
        'status':'ready' if readiness['exit_code']==0 and not readiness['timed_out'] and not readiness['truncated'] else 'quarantined',
        'base_image_reference':recipe.base_image,'install_commands':[],'readiness_command':recipe.readiness_command}
    write_json(output/'recipe.json',recipe_record);write_json(output/'environment.json',built)
    return built
