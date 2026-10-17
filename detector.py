import cv2
from ultralytics import YOLO
from config import Config

cfg = Config()

class Detector:

    def __init__(self, 
                 model_path: str = cfg.model_path, 
                 conf: float = cfg.detection_conn, 
                 device: str = cfg.device
                 ):
        
        self.model_path = model_path
        self.conf = conf
        self.device = device

        self.model = YOLO(self.model_path)


    def findPeople(self, img, draw = True):
        results = self.model(img, conf = self.conf, device = self.device)[0]

        if(draw):
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} : {conf:.2f}"

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

        return img, results