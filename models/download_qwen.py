from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="/root/HybridNER/models/Qwen/",
    local_dir_use_symlinks=False,
    resume_download=True
)
