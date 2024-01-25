import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt")


cap = cv2.VideoCapture("jets_8.mp4")

out = cv2.VideoWriter('Jet Detection.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (int(cap.get(3)), int(cap.get(4))))


while True:
    ret, frame = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.track(frame, conf=0.5)  # only detect car, truck and bus

    boxes = results[0].boxes.xyxy.cpu()
    clss = results[0].boxes.cls.cpu().tolist()

    for box, cls in zip(boxes, clss):
        # cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # calculate center
        cx = int(box[0] + box[2]) // 2
        cy = int(box[1] + box[3]) // 2

        radius = (box[2] - box[0]) / 2
        # Plot center
        cv2.circle(frame, (cx, cy), int(radius), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), int(radius)+2, (255, 255, 255), 2)
        cv2.circle(frame, (cx, cy), int(radius)+4, (0, 255, 0), 2)

        cv2.line(frame, (int(cx), int(cy) - int(radius)), (int(cx), int(cy) + int(radius)), (0, 255, 255), 2)
        cv2.line(frame, (int(cx) - int(radius), int(cy)), (int(cx) + int(radius), int(cy)), (0, 255, 255), 2)

    out.write(frame)

    resized_frame = cv2.resize(frame, (1080, 550))

    cv2.imshow("jets", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()
