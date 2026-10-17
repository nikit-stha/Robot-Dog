from detector import Detector
import cv2
from config import Config

detector = Detector()
cfg = Config()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_heigh)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    
    img, results = detector.findPeople(img)

    print(results)

    cv2.imshow("Tracker", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break