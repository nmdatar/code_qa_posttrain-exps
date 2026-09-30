# Bounded remote validation input

`remote-validation-base.json` is the unchanged former direct-GRPO screen config
from commit `2b70198df1db7b3028448d8f05772e55c52ea6fc`. The
`training_pipeline.remote_validation.prepare` utility derives its one-episode,
$2-capped machinery check from this input; its existing overrides are unchanged.
It is not a recommended standalone training experiment. Tests also use it to
check the remote execution and cost-estimation contract without depending on a
retired campaign directory. Preparing or submitting live validation still needs
the original release and provider setup; no live work is part of repo cleanup.
