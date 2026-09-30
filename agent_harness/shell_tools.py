"""Versioned, read-only shell dialect over the pinned source catalog.

No model text is passed to bash, eval, or a host executable. Commands are parsed
into allowlisted operations and run by the existing sandbox source reader. This
isolates command-string ergonomics from new capabilities and output formatting.
"""
import re
import shlex
from .repository_tools import command as repository_command

VERSION = "source-shell-v1"
PROTOCOL = '''Investigate the pinned repository using one JSON action per turn.
Your only tool is {"tool":"shell","arguments":{"command":"rg -n -F 'symbol'"}}.
Allowed commands (one per call):
  rg --files [-g 'glob'] [--offset N]   list at most 100 source paths
  ls [glob] [--offset N]              list at most 100 source paths
  rg -n -F [-g 'glob'] [--] 'literal text'   literal substring search, at most 100 matches
  cat path                           read the first 120 lines
  head -n N path                     read up to 120 lines
  sed -n 'START,ENDp' path            read a range; at most 120 lines per page
Use repository-relative paths. Quote spaces and glob patterns. Outputs are JSON
with paths, line numbers, hashes, and pagination pointers. Follow next_offset or
next_start_line only if needed. This is a restricted shell dialect: no pipelines,
command chaining, redirects, expansions, environment assignments, arbitrary flags,
other programs, writes, or code execution. Commands inspect pinned source only.
Final action:
{"answer":"Concise answer, at most 120 words", "citations":[{"path":"file.py","start_line":1,"end_line":4}]}.
Cite source lines you actually read. The harness supplies task IDs and source hashes.
Start with a short literal symbol search from the question, then read the matching source.
Do not search a whole natural-language question. Use exact paths returned by tools.
Read small ranges around search matches. Each response must be exactly ONE valid
JSON object, without markdown, prose outside JSON, or a second action.
Within JSON strings, apostrophes need no escaping. Escape double quotes with a backslash.
Keep the answer short and limited to facts supported by the lines you read.
Do not claim code execution. If evidence is insufficient, submit a concise honest
answer; never invent a citation.'''


def parse(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {"command"}:
        raise ValueError("shell requires only a command string")
    text = arguments["command"]
    if not isinstance(text, str) or not text.strip() or len(text.encode()) > 8000:
        raise ValueError("Invalid shell command length")
    # Reject these even inside quotes. This deliberately conservative dialect
    # never needs shell expansion and cannot grow into arbitrary bash execution.
    if any(c in text for c in "\x00\n\r;|&<>`$"):
        raise ValueError("Shell operators and expansions are disabled")
    words = shlex.split(text, comments=False, posix=True)
    if not words:
        raise ValueError("Empty shell command")
    program, args = words[0], words[1:]
    if program == "ls" or (program == "rg" and args[:1] == ["--files"]):
        if program == "rg": args = args[1:]
        result = {}
        while args:
            flag = args.pop(0)
            if flag == "--offset" and args and "offset" not in result:
                value = args.pop(0)
                if not re.fullmatch(r"[0-9]{1,9}", value): raise ValueError("Invalid offset")
                result["offset"] = int(value)
            elif program == "rg" and flag == "-g" and args and "glob" not in result:
                result["glob"] = args.pop(0)
            elif program == "ls" and not flag.startswith("-") and "glob" not in result:
                result["glob"] = "*" if flag == "." else flag
            else:
                raise ValueError("Use rg --files [-g glob] [--offset N] or ls [glob] [--offset N]")
        return "list_files", result
    if program == "rg":
        result = {}; flags = set()
        while args and args[0].startswith("-"):
            flag = args.pop(0)
            if flag == "--": break
            if flag in {"-n", "-F"} and flag not in flags: flags.add(flag)
            elif flag == "-g" and args and "glob" not in result: result["glob"] = args.pop(0)
            else: raise ValueError("Only rg -n -F [-g glob] [--] literal is allowed")
        if len(args) != 1 or flags != {"-n", "-F"} or not args[0]:
            raise ValueError("Use rg -n -F [-g glob] [--] literal")
        return "search_code", {**result, "query": args[0]}
    if program == "cat" and len(args) == 1:
        return "read_file", {"path": args[0], "start_line": 1, "end_line": 120}
    if program == "head" and len(args) == 3 and args[0] == "-n" and re.fullmatch(r"[0-9]{1,9}", args[1]):
        if not 1 <= int(args[1]) <= 120: raise ValueError("head reads 1 to 120 lines")
        return "read_file", {"path": args[2], "start_line": 1, "end_line": int(args[1])}
    if program == "sed" and len(args) == 3 and args[0] == "-n":
        match = re.fullmatch(r"([0-9]{1,9})(?:,([0-9]{1,9}))?p", args[1])
        if match:
            start, end = int(match[1]), int(match[2] or match[1])
            if not 1 <= start <= end: raise ValueError("Invalid line range")
            return "read_file", {"path": args[2], "start_line": start, "end_line": end}
    raise ValueError("Command or flags are outside source-shell-v1 allowlist")


def command(arguments, source_files, workspace_path="/workspace", *, permitted_tools):
    name, translated = parse(arguments)
    if name not in permitted_tools:
        raise ValueError("Command exceeds task source-reading permissions")
    argv = repository_command(name, translated, source_files, workspace_path, paginate_reads=True)
    return argv, name, translated
