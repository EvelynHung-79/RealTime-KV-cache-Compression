from huggingface_hub import snapshot_download
# python scripts/download_model.py

# snapshot_download(
#     repo_id="meta-llama/Llama-2-7b-hf",
#     local_dir="./models/llama2-7b",
# )

# snapshot_download(
#     repo_id="meta-llama/Llama-2-13b-hf",
#     local_dir="./models/llama2-13b",
# )

# snapshot_download(
#     repo_id="meta-llama/Meta-Llama-3-8B",
#     local_dir="./models/llama3-8b",

snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir="./models/llama2-7b-chat",
)