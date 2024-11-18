import cv2

from ultralytics import YOLO

model = YOLO('custom.pt')

# Initialize the camera
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or specify another index for other cameras

# Continuously read frames and predict
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model.predict(source=frame, conf=0.25)

    # Display the frame with predictions
    for result in results:
        annotated_frame = result.plot()  # Annotate the frame with bounding boxes and labels

        # Display in OpenCV window
        cv2.imshow("Live YOLO Prediction", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()