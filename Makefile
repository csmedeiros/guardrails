.PHONY: gen-v3 gen-v2 data-ext prep-ext embed-ext stage3 eval-ext dirs setup setup-vllm data prep embed baseline train-s1 vllm-up gen prep-synth embed-synth stage2 repro-from-synth serve demo mlflow-ui paper

PY = PYTHONPATH=src uv run python
SYN_REPLAY ?= 1.0

dirs:
	mkdir -p data/raw data/processed data/synthetic embeddings mlflow models

setup: dirs
	uv sync

# only needed to (re)generate the synthetic set; needs an NVIDIA GPU
setup-vllm:
	uv venv .venv-vllm --python 3.12
	uv pip install --python .venv-vllm/bin/python vllm==0.33.0

data: dirs
	uv run hf download marcusnogueira/guardbench-ptbr --repo-type dataset --local-dir data/raw/guardbench-ptbr

prep: dirs
	$(PY) src/guardrails/data/prep.py

embed: prep
	$(PY) src/guardrails/data/embed.py data/processed/stage1_en.parquet embeddings/stage1_en_bgem3.npz
	$(PY) src/guardrails/data/embed.py data/processed/stage1_pt.parquet embeddings/stage1_pt_bgem3.npz

baseline: embed
	$(PY) src/guardrails/training/baseline_torch.py baseline_torch_mlp_detailed embeddings/stage1_en_bgem3.npz embeddings/stage1_pt_bgem3.npz

vllm-up:
	.venv-vllm/bin/vllm serve Qwen/Qwen3-4B-AWQ --served-model-name gen --port 8011 \
	  --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager

# optional: needs `make vllm-up` running. Overwrites data/synthetic/synth_v1.parquet (results will differ).
gen:
	NEMO_TELEMETRY_ENABLED=false $(PY) src/guardrails/synth/gen_synth.py 6000

gen-v3:
	NEMO_TELEMETRY_ENABLED=false $(PY) src/guardrails/synth/gen_synth_v3.py 600

gen-v2:
	NEMO_TELEMETRY_ENABLED=false $(PY) src/guardrails/synth/gen_synth_v2.py 6000

prep-synth: dirs
	$(PY) src/guardrails/data/prep_synth.py

embed-synth: prep-synth
	$(PY) src/guardrails/data/embed.py data/processed/synth.parquet embeddings/synth_bgem3.npz

stage2: embed embed-synth
	$(PY) src/guardrails/training/stage2.py $(SYN_REPLAY) final
	cp models/head_final_replay$(SYN_REPLAY).npz models/head.npz

# independent Spanish / multilingual test data + Spanish train split (CC BY 4.0, prompts only)
data-ext: dirs
	uv run hf download ToxicityPrompts/PolyGuardPrompts --repo-type dataset --local-dir data/raw/polyguard
	uv run hf download nvidia/Nemotron-Safety-Guard-Dataset-v3 --repo-type dataset --include "es/*" --local-dir data/raw/nemotron_v3

prep-ext: dirs
	$(PY) src/guardrails/data/prep_ext.py

embed-ext: prep-ext
	for n in pgp_en pgp_pt pgp_es nem_es; do $(PY) src/guardrails/data/embed.py data/processed/$$n.parquet embeddings/$${n}_bgem3.npz; done

# stage 3: add the Nemotron Spanish train split (tag, replay, sources, target recall); needs `make stage2 embed-ext`
stage3: embed-ext
	cp models/head.npz models/head_v1_syn.npz
	$(PY) src/guardrails/training/stage3.py syn_only_r90 1.0 syn 0.90
	$(PY) src/guardrails/training/stage3.py syn_only_r85 1.0 syn 0.85
	$(PY) src/guardrails/training/stage3.py es_nem 1.0 nem 0.95
	$(PY) src/guardrails/training/stage3.py es_syn_nem 1.0 syn,nem 0.95
	$(PY) src/guardrails/training/stage3.py es_syn_nem_r90 1.0 syn,nem 0.90
	$(PY) src/guardrails/training/stage3.py es_syn_nem_r85 1.0 syn,nem 0.85
	cp models/head_es_syn_nem_r90.npz models/head_es.npz

eval-ext:
	$(PY) src/guardrails/evaluation/eval_heads.py models/head_v1_syn.npz models/head_syn_only_r90.npz models/head_syn_only_r85.npz models/head_es_nem.npz models/head_es_syn_nem.npz models/head_es_syn_nem_r90.npz models/head_es_syn_nem_r85.npz

# data + embed + both stages using the shipped synthetic set (no GPU needed for vLLM)
repro-from-synth: data stage2

serve:
	PYTHONPATH=src uv run uvicorn guardrails.serving.serve:app --port 8100

demo:
	uv run python scripts/demo_nemo.py

mlflow-ui:
	uv run mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts --host 127.0.0.1 --port 5000

paper:
	cd paper && tectonic main.tex
