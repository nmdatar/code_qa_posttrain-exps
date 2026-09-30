"""Generous, bounded defaults for interactive demo investigations."""
from agent_harness.contracts import RunLimits

RUN_LIMITS = RunLimits(max_steps=80, max_tool_calls=64, max_output_tokens=64_000,
                      wall_time_seconds=900, max_context_chars=128_000)
CONTEXT_TOKENS = 32_768
MAX_OUTPUT_PER_CALL = 8_192
PROVIDER_TIMEOUT_SECONDS = 180
MAX_REPAIRS = 5
SUPERVISOR_GRACE_SECONDS = 60
MAX_CONCURRENT_RUNS = 8
