# Phase 1 findings (2026-11-02)

## Local environment (measured)
- GPU: RTX 2060, 6 GB (5.3 GB free), compute capability 7.5 (Turing), driver 580.173.02, CUDA 13.0. RAM 31 GB. Disk 272 GB free.
- Python: `uv` venv is 3.13; vllm 0.33.0 (`>=3.11,<3.15`, Apache-2.0) and data-designer 0.9.7 (`>=3.10`, Apache-2.0) both accept it (PyPI JSON, queried today).

## vLLM on Turing (sources are blogs/forums, not official docs)
- CC 7.5 clears the 7.0 floor. Use `--dtype half` (no bf16). FlashAttention-2 needs Ampere+, so a non-FA backend (xFormers/Triton) is used.
- AWQ on sm75 reportedly works again via Marlin kernel PRs (#29901, #31000).
- Reported breakage: Gemma 4 (head_dim=512) overflows Turing shared memory; some newer model backends dropped Turing. **Prefer Qwen-family ~4B AWQ**; avoid Gemma 4.
- **UNVERIFIED on this machine.** Gate for phase 4: install vllm in its own env, serve one 4B AWQ model, check it answers a PT-BR/ES/EN prompt. If it fails, fallback is Ollama/llama.cpp behind the same OpenAI-compatible endpoint.

## NeMo Data Designer
- `pip install data-designer` (0.9.7). Any OpenAI-compatible endpoint works (vLLM listed); endpoint is the API root (`http://localhost:8000/v1`), model name must equal the served name; provider health check runs on first model call. Telemetry off with `NEMO_TELEMETRY_ENABLED=false`.
- Provider config via `data-designer config providers`.

## Datasets and licenses
- `marcusnogueira/guardbench-ptbr`: BeaverTails derivative, EN + PT-BR, 16,188 rows, binary label by 5-LLM vote, no splits/categories/human labels, **CC BY-NC 4.0**, EN/PT label disagreements. No Spanish.
- GuardBench (EMNLP 2024, 40 datasets): library `guardbench` is EUPL-1.2 (PyPI). It includes ES/FR/DE/IT prompt datasets, machine-translated with MADLAD-400-3B-MT. Per-dataset licenses and some datasets need authorization: **not yet checked**; check the library's dataset list before using any ES set.
- BAAI/bge-m3: reported MIT (aggregators; the repo LICENSE file itself not read yet), 1024-dim, 8192 tokens, 100+ languages.

## Resolved
- bge-m3 license: **MIT** (model card header, read directly).
- GuardBench repo docs (`docs/datasets.md`, read directly): all 40 datasets are English. "PromptsES/DE/FR/IT/UnsafeQA will be released soon", so **no Spanish set is available**, contrary to the paper's claim. Some English sets need a form (DecodingTrust, HEx-PHI, HarmEval, TechHazardQA, ToxiGen); licenses are mixed (MIT/Apache/CC BY/CC BY-NC/CC BY-SA).
- Consequence: **Spanish data must be ours**: machine-translate the BeaverTails prompts to ES (local model, hand-spot-check) and generate ES synthetic data. There is no independent ES test set, so the ES test slice is hand-checked, and ES results carry wider confidence intervals.

## Gaps / next
2. vLLM gate test (above).
3. Data prep: splits by prompt id, drop rows where `label_en != label_pt`, report drop rate.
