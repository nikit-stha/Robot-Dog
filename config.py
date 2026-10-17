import torch

class Config:
    detection_conn: float = 0.4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_path: str = "yolo11n.pt"

    frame_width: int = 640
    frame_heigh: int = 480