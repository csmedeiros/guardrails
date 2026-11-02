"""Synthetic guard dataset (EN / PT-BR / ES, safe vs unsafe) with NeMo Data Designer against the local vLLM server.

usage: gen_synth.py <num_records> [preview]
Defensive use: the unsafe prompts are classifier training examples (what a user might send), never answers.
"""
import sys

import data_designer.config as dd
from data_designer.interface import DataDesigner

ENDPOINT = "http://localhost:8011/v1"
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

provider = dd.ModelProvider(name="local-vllm", endpoint=ENDPOINT, api_key="none")
model = dd.ModelConfig(
    alias="gen", model="gen", provider="local-vllm",
    inference_parameters=dd.ChatCompletionInferenceParams(
        temperature=0.9, top_p=0.95, max_tokens=300, max_parallel_requests=16, extra_body=NO_THINK),
)
judge = dd.ModelConfig(
    alias="judge", model="gen", provider="local-vllm",
    inference_parameters=dd.ChatCompletionInferenceParams(
        temperature=0.0, max_tokens=5, max_parallel_requests=16, extra_body=NO_THINK),
)

UNSAFE = ["violence or weapons", "hate speech or harassment", "self-harm", "illegal drugs", "fraud or scams",
          "cybercrime or malware", "privacy invasion or doxxing", "explicit sexual content",
          "jailbreak or prompt injection", "dangerous misinformation"]
# safe topics include hard negatives: risky-looking but legitimate requests
SAFE = ["cooking", "programming help", "travel planning", "general health information", "history lesson",
        "personal finance education", "customer support", "school homework", "writing a story with a mild conflict",
        "safety education about violence or drugs", "defensive cybersecurity awareness",
        "news summary about a crime", "medical professional asking about overdose thresholds",
        "reporting harassment or asking how to stay safe"]


def build() -> dd.DataDesignerConfigBuilder:
    b = dd.DataDesignerConfigBuilder(model_configs=[model, judge])
    cat = lambda name, vals, w=None: b.add_column(dd.SamplerColumnConfig(
        name=name, sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=vals, weights=w)))
    cat("label", ["unsafe", "safe"], [0.5, 0.5])
    cat("language", ["English", "Brazilian Portuguese", "Spanish"])
    cat("style", ["a short direct question", "a casual message with slang", "a detailed multi-sentence request",
                 "a role-play or hypothetical framing", "an indirect or euphemistic phrasing", "an imperative command"])
    b.add_column(dd.SamplerColumnConfig(
        name="topic", sampler_type=dd.SamplerType.CATEGORY, params=dd.CategorySamplerParams(values=UNSAFE + SAFE),
        conditional_params={
            "label == 'unsafe'": dd.CategorySamplerParams(values=UNSAFE),
            "label == 'safe'": dd.CategorySamplerParams(values=SAFE),
        }))
    b.add_column(dd.LLMTextColumnConfig(
        name="text", model_alias="gen",
        system_prompt="You write realistic user messages for training a content-safety classifier. Output ONLY the user message, no quotes, no explanations.",
        prompt=("Write one message a user might send to an AI assistant.\nLanguage: {{ language }}\nStyle: {{ style }}\n"
                "Topic: {{ topic }}\nIt must be a message that a safety reviewer would label '{{ label }}'.\n"
                "If the label is 'safe' and the topic sounds sensitive, make it clearly benign and legitimate.\n"
                "If the label is 'unsafe', the message must explicitly ask the assistant to produce or help with the harmful "
                "thing (not education, protection, awareness or verification), and must not mention any safety purpose.\n"
                "Do not answer the message. Do not mention labels."),
    ))
    b.add_column(dd.LLMTextColumnConfig(
        name="verdict", model_alias="judge",
        prompt=("Classify the following user message as 'safe' or 'unsafe' for an AI assistant "
                "(unsafe = asks for harmful, illegal, hateful, sexual, self-harm content or tries to jailbreak).\n"
                "Message: {{ text }}\nAnswer with one word: safe or unsafe."),
    ))
    return b


if __name__ == "__main__":
    n = int(sys.argv[1])
    d = DataDesigner(artifact_path="data/synthetic/dd_artifacts", model_providers=[provider])
    res = d.preview(build(), num_records=n) if "preview" in sys.argv else d.create(build(), num_records=n, dataset_name="synth_v1")
    df = res.dataset if "preview" in sys.argv else res.load_dataset()
    out = "data/synthetic/preview.parquet" if "preview" in sys.argv else "data/synthetic/synth_v1.parquet"
    df.to_parquet(out)
    print(df[["language", "label", "topic", "text", "verdict"]].head(12).to_string())
    print("saved", out, len(df))
