import cv2

# Change the source to '0' for the default camera, or use the video file path
cap = cv2.VideoCapture('/dev/video0')

if not cap.isOpened():
    print("Error: Could not open video source.")
else:
    print("Video source is open.")
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab a frame")
    else:
        print("Frame grabbed successfully")
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

cap.release()
