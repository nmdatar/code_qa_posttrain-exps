"""Repository tool adapters. No private references are exposed to the solver."""
import json
from pathlib import Path


class InfrastructureError(RuntimeError):
    pass


class InvalidToolCall(ValueError):
    pass


class FakeEnvironment:
    def __init__(self, files=None, allowed_tools=None):
        self.files = files or {'README.md': 'Example source for offline integration tests.'}
        self.allowed_tools = allowed_tools or ['list_files', 'search_code', 'read_file']

    def call(self, name, args):
        if name not in self.allowed_tools:
            raise InvalidToolCall('Tool not permitted')
        if name == 'list_files':
            return {'files': sorted(self.files)}
        if name == 'read_file':
            if args.get('path') not in self.files:
                raise InvalidToolCall('Unknown file')
            return {'path': args['path'], 'lines': self.files[args['path']].splitlines()}
        if name == 'search_code':
            return {'matches': [p for p, text in self.files.items() if args.get('query', '') in text]}
        raise InvalidToolCall('Unknown tool')


class ModalEnvironment:
    def __init__(self, bundle, task_id, output_dir, budget=None, call_upper_usd=None):
        from dataset_builder.investigate import start
        self.output_dir = Path(output_dir)
        self.record = start(bundle, task_id, output_dir)
        if self.record['backend'] != 'modal':
            raise ValueError('Modal environment requires Modal-built bundle')
        self.allowed_tools = self.record['task']['permitted_tools']
        self.budget, self.call_upper_usd = budget, call_upper_usd

    def call(self, name, args):
        from dataset_builder.investigate import tool, _command
        # Separate deterministic argument failures from provider failures.
        try:
            _command(self.record, name, args)
        except (ValueError, TypeError, KeyError) as exc:
            raise InvalidToolCall(str(exc)) from exc
        if self.budget is None or self.call_upper_usd is None or self.call_upper_usd <= 0:
            raise InfrastructureError('Modal tool call requires explicit spending reservation')
        event = self.budget.execute('modal', self.call_upper_usd,
            lambda: tool(self.output_dir, name, args), {'operation': name})
        if event['status'] == 'error':
            raise InfrastructureError(event.get('error', 'Repository tool infrastructure failed'))
        return {'status': event['status'], 'observation': event['observation']}
