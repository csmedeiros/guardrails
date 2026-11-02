@AGENTS.md

## Claude Code specifics

- Skills in `.claude/skills/`: use `reproduce` to rebuild the experiments or the model, and `ask-guardrails` to answer questions
  about the project, data, results or API from the repository evidence.
- Long-running commands (`make vllm-up`, `make serve`, `make mlflow-ui`, `make gen`) must run in the background; wait on a
  condition (health endpoint, log line) rather than sleeping.
- Prefer the Read/Edit tools for files; keep temporary files out of the repo.
- **Evals need an approved plan first** (see AGENTS.md "Present the test plan"). Do not launch any eval, benchmark, judge run or
  paid API loop, in the foreground or background, until the user has said yes to the plan you showed. Starting several models
  or several datasets "to be thorough" without approval is a violation. Use AskUserQuestion only when a decision is open;
  otherwise show the plan and wait.

- Guard models are chosen in `configs/guard_models.yaml` (`GUARD_MODEL`); the HF token is in `.env` (git-ignored), never print it. Do not discard the training pipeline when changing how the model is distributed.
