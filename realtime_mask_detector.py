import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("face_mask_model.keras")

# Load OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

IMG_SIZE = 100

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        prediction = model.predict(face_array, verbose=0)
        confidence = prediction[0][0]

        if confidence > 0.5:
            label = f"No Mask ❌ ({confidence*100:.2f}%)"
            color = (0, 0, 255)  # Red
        else:
            label = f"Mask ✅ ({(1-confidence)*100:.2f}%)"
            color = (0, 255, 0)  # Green

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Mask Detector", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
