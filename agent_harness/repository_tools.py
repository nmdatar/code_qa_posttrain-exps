"""Trusted host dispatcher. Repository reads run inside the disposable sandbox."""
import json

# Sent as an argv element, never interpolated into a shell. No host data is mounted.
READ_SCRIPT = r'''
import fnmatch, hashlib, json, pathlib, sys
root=pathlib.Path('/repo').resolve()
request=json.loads(sys.argv[1]); name=request['tool']; a=request['arguments']
files=json.loads(''.join(sys.argv[2:]))
def path(name):
 p=root/name
 if name not in files or p.is_symlink() or any(x.is_symlink() for x in p.parents if x!=root and x.is_relative_to(root)) or not p.resolve().is_relative_to(root):
  raise ValueError('Not a regular tracked repository file')
 if p.stat().st_size>2000000: raise ValueError('File exceeds read limit')
 return p
if name=='list_files':
 matches=[f for f in files if fnmatch.fnmatch(f,a.get('glob','*'))]
 offset=a.get('offset',0)
 print(json.dumps({'files':matches[offset:offset+100],'next_offset':offset+100 if len(matches)>offset+100 else None}))
elif name=='read_file':
 p=path(a['path']); raw=p.read_bytes(); lines=raw.decode().splitlines(); start=a.get('start_line',1); end=min(a.get('end_line',start+119),len(lines))
 if start<1 or end<start: raise ValueError('Invalid line range')
 page_end=min(end,start+119) if request.get('paginate_reads',False) else end
 if page_end-start>=120: raise ValueError('Invalid line range; read at most 120 lines')
 next_line=page_end+1 if page_end<end else None
 end=page_end
 print(json.dumps({'path':a['path'],'start_line':start,'end_line':end,'file_sha256':hashlib.sha256(raw).hexdigest(),**({'next_start_line':next_line} if request.get('paginate_reads',False) else {}),'lines':[str(i)+': '+lines[i-1] for i in range(start,end+1)]}))
elif name=='search_code':
 found=[]
 for f in files:
  if not fnmatch.fnmatch(f,a.get('glob','*')): continue
  try: lines=path(f).read_text().splitlines()
  except (ValueError,UnicodeError,OSError): continue
  for i,line in enumerate(lines,1):
   if a['query'] in line: found.append({'path':f,'line':i,'text':line[:500]})
   if len(found)>=100: break
  if len(found)>=100: break
 print(json.dumps({'matches':found,'limit':100}))
'''


def command(name, arguments, source_files=(), workspace_path="/repo", *, paginate_reads=False):
    if workspace_path not in {"/repo", "/workspace"}:
        raise ValueError("Unsupported workspace")
    specs = {'list_files': ({}, {'glob': str, 'offset': int}),
             'search_code': ({'query': str}, {'glob': str}),
             'read_file': ({'path': str}, {'start_line': int, 'end_line': int}),
             'python_probe': ({'code': str}, {})}
    if name not in specs or not isinstance(arguments, dict):
        raise ValueError('Unknown tool or invalid arguments')
    required, optional = specs[name]
    if set(required) - set(arguments) or set(arguments) - set(required) - set(optional):
        raise ValueError('Missing or unknown tool arguments')
    for k, value in arguments.items():
        if type(value) is not {**required, **optional}[k]:
            raise ValueError('Invalid argument type')
        if isinstance(value, str) and (len(value.encode()) > 8000 or '\0' in value):
            raise ValueError('Argument exceeds limit')
        if type(value) is int and value < 0:
            raise ValueError('Negative offset')
    if name == 'python_probe':
        return ['python3', '-c', arguments['code']]
    files = sorted(source_files)
    if name == 'read_file':
        if arguments['path'] not in files:
            raise ValueError('Path is not in the pinned source catalog')
        files = [arguments['path']]
    catalog = json.dumps(files)
    if len(catalog.encode()) > 1_000_000:
        raise ValueError('Source catalog exceeds command metadata limit')
    # git archive images deliberately contain no .git. Supply only the public
    # catalog from the pristine host checkout, split below Linux's argv limit.
    chunks = [catalog[i:i+32000] for i in range(0, len(catalog), 32000)]
    return ['python3', '-c', READ_SCRIPT.replace("pathlib.Path('/repo')", 'pathlib.Path(' + repr(workspace_path) + ')'), json.dumps({'tool': name, 'arguments': arguments, **({'paginate_reads':True} if paginate_reads else {})}), *chunks]
