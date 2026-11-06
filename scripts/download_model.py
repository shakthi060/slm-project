from pathlib import Path

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    local_dir=str(Path(r"D:\shakthi\slm-artifacts\models\Qwen2.5-1.5B-Instruct")),
    max_workers=8,
)
print("DONE")
