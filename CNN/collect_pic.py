#collect_pic.py
import cv2
import os

# 1. Setup
name = input("Enter your ID (e.g., 22-48108-2): ")
save_path = f"CNN/dataset/{name}"

# Create folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)
    current_count = 0
else:
    # Check existing files so we don't overwrite them
    existing = [f for f in os.listdir(save_path) if f.endswith('.jpg')]
    current_count = len(existing)

print(f"Storing images in: {save_path}")
print(f"Current images: {current_count}")
print("Press 's' to save a photo. Press 'q' to quit.")

# 2. Camera Setup (Mac Optimized)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret: break

    # Create a copy for saving (clean, no boxes drawn)
    save_frame = frame.copy()

    # Detection for visual feedback
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw box so you know you are in frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display info
    cv2.putText(frame, f"Saved: {current_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # 3. Save Logic
    if key == ord('s'):
        # We save the WHOLE frame. The generator will handle cropping/resizing later.
        # OR: If you want to crop the face specifically:
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            # Add some padding so we don't cut the chin
            p = 20 
            face_img = save_frame[max(0,y-p):min(y+h+p, frame.shape[0]), max(0,x-p):min(x+w+p, frame.shape[1])]
            
            filename = f"{save_path}/new_{current_count}.jpg"
            cv2.imwrite(filename, face_img)
            print(f"Saved {filename}")
            current_count += 1
        else:
            print("No face detected! Cannot save.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()