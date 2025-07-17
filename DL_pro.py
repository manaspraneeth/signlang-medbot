import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from collections import deque
import pyttsx3
from openai import AzureOpenAI

# === Load ASL model ===
model = load_model('my_model.keras')

# === Load class labels ===
with open('class_indices.pkl', 'rb') as f:
    class_indices = pickle.load(f)
labels = [None] * len(class_indices)
for label, idx in class_indices.items():
    labels[idx] = label

# === Azure OpenAI Setup ===
client = AzureOpenAI(
    api_key="1Zz65AexHtUXdqRnZB1Nm79eHICIDk4VV1jVDQ8dw36u0C8QsWKWJQQJ99BFACHYHv6XJ3w3AAAAACOG83X6",
    api_version="2025-01-01-preview",
    azure_endpoint="https://amen-mbsyxyt9-eastus2.cognitiveservices.azure.com/"
)

# === Webcam and UI Setup ===
cap = cv2.VideoCapture(0)
IMG_SIZE = 64
pred_queue = deque(maxlen=5)
recorded_text = ""
current_letter = ""
font = cv2.FONT_HERSHEY_SIMPLEX
tts_engine = pyttsx3.init()

print("Press SPACE to record the current letter, ENTER to ask medical question, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Define ROI ===
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = roi_resized.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # === Predict ASL Gesture ===
    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction)
    pred_queue.append(class_id)
    most_common_pred = max(set(pred_queue), key=pred_queue.count)
    class_label = labels[most_common_pred]
    current_letter = class_label

    # === Draw Interface ===
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Letter: {current_letter}", (x1, y1 - 10), font, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {recorded_text}", (50, 400), font, 1, (255, 255, 255), 2)

    cv2.imshow("ASL Recognition", frame)
    cv2.imshow("ROI", roi_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # SPACE to capture letter
        recorded_text += current_letter
        print(f"Appended: {current_letter}")
    elif key == 13:  # ENTER to submit query
        print(f"Sending to medical chatbot: {recorded_text}")
        try:
            response = client.chat.completions.create(
                model="gpt-35-turbo",  # Deployment name
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": recorded_text}
                ]
            )
            reply = response.choices[0].message.content
            print("Chatbot response:", reply)
            tts_engine.say(reply)
            tts_engine.runAndWait()
        except Exception as e:
            print("Chatbot error:", str(e))
        recorded_text = ""
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()