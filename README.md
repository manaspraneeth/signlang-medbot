# signlang-medbot

AI-powered chatbot that translates ASL (American Sign Language) gestures into medical queries for hearing-impaired users.  
Uses deep learning (MobileNetV2) for gesture recognition and provides real-time text/audio responses to improve healthcare accessibility.

👉 **Download the pre-trained model** from this link:  
[my_model.keras – Google Drive](https://drive.google.com/file/d/1HFw5d2yWKYgtooXS-V2R4T6iJ0T47Npd/view?usp=sharing)  
Place it in the root directory as: `my_model.keras`


## 🔗 Dataset

The model was trained on the ASL Alphabet dataset.

👉 **Download the dataset** from:  
[ASL Dataset – Google Drive](https://drive.google.com/file/d/1r9JAW_QIjk9OV-wzviOUfUzhywp-mzfs/view?usp=sharing)  
After downloading, extract and place it in the `dataset/` folder in the root directory.


DEMO:

<p align="center">
  <img src="https://github.com/user-attachments/assets/b8e23265-caa9-436e-a55e-c678dc09efc5" width="400"/>
  <img width="403" height="623" alt="image" src="https://github.com/user-attachments/assets/885a19b6-5ce5-43b4-a634-ffd0f71c1714" />
  <img width="417" height="560" alt="image" src="https://github.com/user-attachments/assets/62348f88-6289-4489-8d89-6e3abc659764" />
  <img width="409" height="557" alt="image" src="https://github.com/user-attachments/assets/a9fe5888-73fb-4c9d-aa2b-3d7e64422704" />



</p>

**How it works:**
1. User shows an ASL gesture within the green region of interest (ROI).
2. The system recognizes the gesture using a deep learning model (MobileNetV2).
3. Each recognized letter is added to form a word or query.
4. Pressing `ENTER` sends the text query to the chatbot.
5. The chatbot returns a real-time medical response (shown on screen and spoken out loud).

*Goal:* Enable real-time medical communication for hearing-impaired users using computer vision and AI.
