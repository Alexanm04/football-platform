from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="martinjolif/yolo-football-player-detection",
    filename="yolo-football-player-detection.pt"
)
print(path)