import cv2
from ultralytics import YOLO


def main():
    cap = cv2.VideoCapture(0)
    model = YOLO("yolov8n-pose.pt")
    while True:
        success, frame = cap.read()
        if success:
            result = model(frame)[0]
            annotated_frame = result.plot()
            cv2.imshow("Live", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break


if __name__ == '__main__':
    main()
