"""Shared synchronous model/action/observation loop for training and evaluation."""
from dataclasses import asdict
import json
import time
from training_pipeline.contracts import InfrastructureError, Trajectory, VerificationResult, PolicyFormatError
from training_pipeline.budget import BudgetLimit
from training_pipeline.storage import digest, atomic_json
from .modal_backend import BudgetExceeded, SandboxInfrastructureError


def observation_text(observation, cap):
    """Preserve every parallel call identity when trimming the shared turn budget."""
    import copy
    value = copy.deepcopy(observation)
    if isinstance(value, dict) and 'tool_results' in value:
        for item in value['tool_results']:
            item.pop('arguments', None)  # Full request remains in the durable trajectory.
        while len(json.dumps(value, ensure_ascii=True).encode()) > cap:
            candidates = [(len(item['observation'].get(key, '')), item['observation'], key)
                          for item in value['tool_results'] for key in ('stdout', 'stderr')
                          if item['observation'].get(key)]
            if not candidates:
                break
            _, result, key = max(candidates, key=lambda entry: entry[0])
            result[key] = result[key][:len(result[key]) // 2]
            result[key + '_truncated'] = True
    text = json.dumps(value, ensure_ascii=True)
    if len(text.encode()) > cap:
        text = text.encode()[:cap].decode('utf-8', errors='ignore') + '\n[output truncated]'
    return text


def run_episode(backend, factory, task, limits, *, run_id, stage, group_id, episode_id,
                experiment_hash, temperature, tracker):
    trajectory = Trajectory(run_id, stage, task['id'], digest(task), group_id, episode_id,
        backend.policy_id, factory.identity, experiment_hash, task['split'])
    start = time.monotonic()
    timings = {}
    def measured(phase, call):
        began = time.monotonic()
        try:
            return call()
        finally:
            timings[phase] = timings.get(phase, 0) + time.monotonic()-began
    episode = None
    output_tokens = 0
    tool_calls = 0
    tool_turns = 0
    max_parallel_calls = 0
    try:
        episode = measured('provision_seconds', lambda: factory.create(task, episode_id, tracker.root / 'trajectories' / (episode_id + '.json')))
        # Task-specific limits can only narrow the run limits.
        effective = dict(limits)
        for k, v in getattr(episode, 'limits', {}).items():
            if k in effective:
                effective[k] = min(effective[k], v)
        trajectory.events.append({'kind': 'initial', 'messages': episode.messages, 'limits': effective,
                                  'model_identity': backend.identity})
        for _ in range(effective['max_generations']):
            if time.monotonic() - start >= effective['latency_seconds'] or output_tokens >= effective['max_output_tokens']:
                trajectory.termination = 'budget_exhausted'
                trajectory.events.append({'kind':'termination_cause','reason':
                    'output_tokens' if output_tokens >= effective['max_output_tokens'] else 'latency'})
                break
            max_tokens = min(effective['max_tokens_per_call'], effective['max_output_tokens'] - output_tokens)
            if hasattr(episode, 'prepare_generation'):
                episode.remaining_tool_calls = effective['max_tool_calls'] - tool_calls
                episode.prepare_generation(effective['max_generations'] - _)
            generation = measured('generation_seconds', lambda: backend.sample(episode.messages, max_tokens, temperature))
            generation.validate()
            if generation.policy_id != trajectory.policy_id:
                raise ValueError('Policy changed within episode')
            trajectory.generations.append(generation)
            output_tokens += len(generation.tokens)
            trajectory.events.append({'kind': 'generation', **asdict(generation)})
            if hasattr(episode, 'usage'):
                episode.usage(len(generation.prompt), len(generation.tokens))
            episode.messages.append({'role': 'assistant', 'content': generation.text})
            # Persist before tool execution, so a lost tool response does not lose the action.
            atomic_json(tracker.root / 'trajectories' / (episode_id + '.json'), trajectory.to_dict())
            if time.monotonic() - start >= effective['latency_seconds']:
                trajectory.termination = 'budget_exhausted'
                break
            try:
                action = episode.parse_action(generation.text) if hasattr(episode, 'parse_action') else json.loads(generation.text)
                if hasattr(episode, 'parse_action'):
                    trajectory.events.append({'kind': 'parsed_action', 'value': action})
                if not isinstance(action, dict):
                    raise ValueError('Action must be a JSON object')
                count = (episode.action_tool_count(action) if hasattr(episode, 'action_tool_count')
                         else int('tool' in action))
                if count:
                    if tool_calls + count > effective['max_tool_calls']:
                        raise BudgetExceeded('Tool budget exhausted')
                    tool_calls += count
                    tool_turns += 1
                    max_parallel_calls = max(max_parallel_calls, count)
                done, observation = measured('action_seconds', lambda: episode.step(action))
            except (ValueError, KeyError, TypeError) as exc:
                is_format_failure = isinstance(exc, (json.JSONDecodeError, PolicyFormatError))
                if is_format_failure:
                    trajectory.events.append({'kind':'format_failure','error_type':type(exc).__name__})
                if getattr(episode, 'invalid_action_policy', 'retry-v1') == 'zero-v1' and is_format_failure:
                    trajectory.termination = 'invalid_format'
                    break
                done, observation = False, {'error': 'Invalid action: ' + type(exc).__name__,
                    'detail': str(exc)[:500], 'instruction': 'Correct the action. Return exactly one JSON object.'}
            trajectory.events.append({'kind': 'observation', 'value': observation})
            if done:
                trajectory.submission = observation
                trajectory.termination = 'completed'
                break
            text = observation_text(observation, effective['max_tool_output_bytes'])
            episode.messages.append({'role': 'user', 'content': 'Tool observation: ' + text})
            if hasattr(episode, 'remember_observation'):
                episode.remember_observation(observation)
        else:
            trajectory.termination = 'budget_exhausted'
            trajectory.events.append({'kind':'termination_cause','reason':'generations'})
        trajectory.verification = measured('verification_seconds', lambda: (
            episode.verify_invalid_format(trajectory) if trajectory.termination == 'invalid_format'
            else episode.verify(trajectory)))
        trajectory.verification.validate()
    except BudgetLimit:
        trajectory.verification = VerificationResult('unresolved', None, factory.reward_version, ['spending ceiling'])
        raise
    except (InfrastructureError, SandboxInfrastructureError, TimeoutError, OSError) as exc:
        trajectory.termination = 'infrastructure_error'
        trajectory.verification = VerificationResult('unresolved', None, factory.reward_version,
                                                     [type(exc).__name__], retryable=True)
    except BudgetExceeded:
        trajectory.termination = 'budget_exhausted'
        trajectory.verification = measured('verification_seconds', lambda: episode.verify(trajectory)) if episode else VerificationResult(
            'unresolved', None, factory.reward_version, ['No episode'])
    except ValueError as exc:
        trajectory.termination = 'agent_error'
        # Invalid provider token records cannot be trained as a policy outcome.
        trajectory.verification = VerificationResult('unresolved', None, factory.reward_version, [str(exc)])
    finally:
        if episode:
            try:
                measured('cleanup_seconds', episode.close)
            except Exception as exc:
                trajectory.termination = 'infrastructure_error'
                trajectory.verification = VerificationResult('unresolved', None, factory.reward_version,
                    ['Cleanup failed: ' + type(exc).__name__], retryable=True)
        trajectory.usage = {'input_tokens': sum(len(g.prompt) for g in trajectory.generations),
                            'output_tokens': output_tokens, 'tool_calls': tool_calls,
                            'tool_turns': tool_turns, 'max_parallel_calls': max_parallel_calls,
                            'mean_calls_per_tool_turn': tool_calls / tool_turns if tool_turns else 0.0,
                            'latency_seconds': time.monotonic() - start, 'cost_usd': None, **timings}
        recorder = getattr(episode, 'recorder', None)
        if recorder is not None and 'tool_seconds' in recorder.record:
            trajectory.usage['tool_seconds'] = recorder.record['tool_seconds']
        tracker.trajectory(trajectory)
    return trajectory
