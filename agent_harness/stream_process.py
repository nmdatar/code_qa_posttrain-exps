"""Sequential sandbox stdin/stdout transport for prebuilt nonroot source images."""
import json
import queue
import threading
import time
from .process import CommandResult

SERVER = r'''
import json, subprocess, sys, tempfile, time
for line in sys.stdin:
 request=json.loads(line)
 start=time.monotonic()
 with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
  try:
   p=subprocess.run(request['argv'], stdout=out, stderr=err, timeout=request['timeout'])
   code=p.returncode
  except subprocess.TimeoutExpired:
   code=-1
  size_out=out.tell();size_err=err.tell();out.seek(0);err.seek(0)
  cap=request['cap']
  result={'stdout':out.read(cap).decode('utf-8','replace'),'stderr':err.read(cap).decode('utf-8','replace'),
          'exit_code':code,'elapsed_seconds':time.monotonic()-start,'stdout_bytes':size_out,'stderr_bytes':size_err,
          'stdout_truncated':size_out>cap,'stderr_truncated':size_err>cap}
 print(json.dumps(result),flush=True)
'''


class StreamProcess:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.lines = queue.Queue()
        self.started = False

    def _read(self):
        buffer = ''
        try:
            for chunk in self.sandbox.stdout:
                buffer += chunk.decode() if isinstance(chunk,bytes) else chunk
                if len(buffer) > 1_000_000:
                    raise ValueError('Sandbox protocol response exceeds bound')
                while '\n' in buffer:
                    line,buffer = buffer.split('\n',1)
                    self.lines.put(json.loads(line))
            self.lines.put(RuntimeError('Sandbox command server exited'))
        except BaseException as exc:
            self.lines.put(exc)

    def run(self, argv, timeout, cap):
        if not self.started:
            threading.Thread(target=self._read,daemon=True).start()
            self.started = True
        def send():
            try:
                value = json.dumps({'argv':list(argv),'timeout':timeout,'cap':cap})+'\n'
                for offset in range(0,len(value),32000):
                    self.sandbox.stdin.write(value[offset:offset+32000])
                    self.sandbox.stdin.drain()
            except BaseException as exc:
                self.lines.put(exc)
        threading.Thread(target=send,daemon=True).start()
        try:
            value = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError('Sandbox stream command exceeded deadline') from None
        if isinstance(value,BaseException):
            raise value
        result = CommandResult(**value)
        if result.exit_code < 0:
            raise TimeoutError('Sandbox command timed out')
        return result
