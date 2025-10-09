from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-hf",
    local_dir="./models/llama2-7b",
    token="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"  # 替換為你的 token
)

# snapshot_download(
#     repo_id="meta-llama/Llama-2-13b-hf",
#     local_dir="./models/llama2-13b",
#     token="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"  # 替換為你的 token
# )

# snapshot_download(
#     repo_id="meta-llama/Meta-Llama-3-8B",
#     local_dir="./models/llama3-8b",
#     token="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"  # 替換為你的 token
# )
