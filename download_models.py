import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = "/data/zikun_workspace/.cache"
os.environ["HF_TOKEN"] = "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"
from huggingface_hub import snapshot_download


HF_HOME = "/data/zikun_workspace/.cache"
MODEL_ROOT = "/data/model_weights_public"

MODEL_IDS = [
    "StanfordShahLab/llama-base-4096-clmbr"
]


def main():
    for model_id in MODEL_IDS:
        snapshot_download(
            repo_id=model_id,
            local_dir=f"{MODEL_ROOT}/{model_id}",
        )


if __name__ == "__main__":
    main()
