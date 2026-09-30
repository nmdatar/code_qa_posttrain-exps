"""Unrestricted bash command construction for disposable repository sandboxes."""

PROTOCOL = '''Investigate the pinned repository using one JSON action per turn. The only tool is:
{"tool":"bash","arguments":{"command":"ls"}}.
Commands run in bash in /workspace, in a fresh repository sandbox with no network or credentials.
You may run arbitrary commands, pipelines, scripts, and programs available in the sandbox. There is no command or path allowlist.
Use shell commands to find relevant source and read numbered lines (for example grep -R -n, find, or nl -ba file.py | sed -n '10,40p').
Tool output and episode budgets still apply. Final action:
{"answer":"Concise answer, at most 120 words", "citations":[{"path":"file.py","start_line":1,"end_line":4}]}.
Cite original repository source lines you actually read, using paths relative to /workspace.
The harness supplies task IDs and hashes from the pinned original source; edits do not change the source used for grading.
Each response must be exactly ONE valid JSON object, without markdown, prose outside JSON, or a second action.
Within JSON strings, apostrophes need no escaping. Escape double quotes with a backslash.
Keep the answer short and limited to facts supported by the lines you read. Only claim execution when supported by tool output.
If evidence is insufficient, submit a concise honest answer; never invent a citation.'''


def command(arguments):
    if not isinstance(arguments, dict) or set(arguments) != {'command'} or not isinstance(arguments['command'], str):
        raise ValueError('bash requires one string command')
    value = arguments['command']
    if not value.strip() or '\0' in value:
        raise ValueError('bash command must be nonempty and contain no NUL')
    return ['bash', '-lc', value]
