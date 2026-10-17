from detector import Detector
import cv2

detector = Detector()

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    
    img, results = detector.findPeople(img)

    cv2.imshow("Tracker", img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break