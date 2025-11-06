
Facial-Emotion-Recognition-Using-CNN


 😊 Facial Emotion Recognition Using CNN

This project focuses on building a **Facial Emotion Recognition (FER)** system using **Convolutional Neural Networks (CNNs)**.
The model is trained on a **FER-2013-like dataset** containing labeled grayscale images representing seven basic human emotions.
The main goal is to automatically classify human facial expressions into emotion categories such as *Happy, Sad, Angry, Fear, Disgust, Surprise,* and *Neutral.*



📂 Project Structure


Facial-Emotion-Recognition-Using-CNN/
│
├── dataset/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   └── surprise/
│   ├── test/
│       ├── (same folders as train)
│
├── notebooks/
│   └── ImageClassification.ipynb      # Jupyter Notebook with full code
│
├── models/
│   └── emotion_classifier_final.keras # Saved trained model
│
├── results/
│   ├── sample_predictions/            # Output prediction images
│   └── training_plots.png             # Accuracy/Loss curves
│
├── README.md
└── requirements.txt




⚙️ Tech Stack

* Language: Python
* Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, OpenCV
* Environment: Jupyter Notebook / Google Colab



📊 Dataset Description

The dataset is inspired by FER-2013, containing grayscale facial images (48×48 pixels).
Each image is labeled under one of seven emotion classes:

  
  Angry 😡
  Disgust 😖
  Fear 😨
  Happy 😄
  Neutral 😐
  Sad 😢
  Surprise 😲
  
 Data is divided into training and testing sets for supervised learning.



## 🧩 **Project Workflow**

| Step                       | Description                                                                                                   |
| -------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 1. Dataset Acquisition** | Loaded and organized FER-2013-like dataset with emotion labels.                                               |
| 2. Data Preprocessing**  | Resized all images to (48×48), normalized pixel values, and applied data augmentation (rotation, flip, zoom). |
| 3. Model Design**        | Built a **Custom CNN** architecture with convolutional, pooling, dropout, and dense layers.                   |
| 4. Model Compilation**   | Used **Adam optimizer**, **categorical crossentropy** loss, and **accuracy** as a metric.                     |
| 5. Model Training**      | Trained for 30 epochs using callbacks like **EarlyStopping** and **ReduceLROnPlateau** to avoid overfitting.  |
| 6. Evaluation**          | Tested on unseen data — achieved around **69% training accuracy** and **64% test accuracy**.                  |
| 7. Prediction**          | The trained model successfully predicted emotions from new unseen images with up to **98% confidence**.       |
| 8. Model Saving**        | Saved final model as `emotion_classifier_final.keras` for future deployment.                                  |



 📈 Model Performance

| Metric              | Value  |
| ------------------- | ------ |
| Training Accuracy   | 0.69   |
| Validation Accuracy | 0.64   |
| Test Accuracy       | 0.6393 |
| Test Loss           | 0.9858 |

✅ The model shows consistent results with minimal overfitting, demonstrating effective learning of emotion-based facial features.



 🖼️ Sample Predictions

Example model outputs:

| Image | Predicted Emotion | Confidence |
| ----- | ----------------- | ---------- |
| 🙂    | Happy             | 98.58%     |
| 😡    | Angry             | 97.34%     |
| 😢    | Sad               | 95.12%     |
| 😲    | Surprise          | 96.40%     |

---

💾 Model Saving & Loading

python
 Save model
model.save("emotion_classifier_final.keras")

 Load model
from tensorflow.keras.models import load_model
model = load_model("emotion_classifier_final.keras")

 Evaluate
test_loss, test_acc = model.evaluate(test_generator)
print(f"✅ Test Accuracy: {test_acc:.4f}")




🚀 Future Enhancements

* Deploy the model using **Streamlit or Flask** for real-time facial emotion recognition.
* Integrate **OpenCV webcam feed** for live emotion detection.
* Fine-tune with a **pre-trained model (MobileNetV2 / VGG16)** to improve accuracy.
* Implement **multi-face emotion detection** in a single frame.



🧠 Key Learnings

* Improved understanding of CNN architecture and transfer learning.
* Hands-on experience with image preprocessing, augmentation, and model optimization.
* Explored real-world emotion recognition use cases in AI and Computer Vision.





Would you like me to create a short **one-line GitHub repo description** (the small tagline under the repo title) too?
For example:

> “Facial Emotion Recognition using CNN — Detecting human emotions from facial expressions with deep learning.”
