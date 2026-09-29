"""Validated JSON defaults for the evaluation-only command."""
import json
import os
from pathlib import Path
import re


def apply_config(parser, argv):
    probe, _ = parser.parse_known_args(argv)
    if probe.config is None:
        return
    path = probe.config.resolve()
    try:
        values = json.loads(path.read_text())
    except (OSError, ValueError):
        parser.error("Cannot read evaluation config JSON")
    if not isinstance(values, dict):
        parser.error("Evaluation config must be a JSON object")
    actions = {a.dest: a for a in parser._actions if a.dest not in {'help', 'config'}}
    supplied = {arg.split('=', 1)[0] for arg in argv if arg.startswith('--')}
    defaults = {}
    for key, value in values.items():
        if key not in actions:
            parser.error(f"Unknown evaluation config field: {key}")
        action = actions[key]
        if supplied.intersection(action.option_strings):
            continue
        if isinstance(value, str):
            match = re.fullmatch(r'\$\{([A-Za-z_][A-Za-z_0-9]*)\}', value)
            if match:
                name = match[1]
                value = os.environ.get(name)
                if not value:
                    parser.error(f"Set environment variable {name} for config field {key}")
        if key == 'tags':
            valid = isinstance(value, list) and all(isinstance(x, str) and x.strip() for x in value)
        elif action.type is int:
            valid = type(value) is int
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            parser.error(f"Invalid type or empty value for config field {key}")
        if action.choices and value not in action.choices:
            parser.error(f"Unsupported value for config field {key}")
        if action.type is Path:
            value = Path(value).expanduser()
            if not value.is_absolute():
                value = (path.parent / value).resolve()
        defaults[key] = value
    parser.set_defaults(**defaults)
