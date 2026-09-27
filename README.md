# 🇪🇹 Ethiopian Birr Banknote Denomination Recognition

A deep learning-powered web application that automatically identifies Ethiopian Birr banknote denominations from images. Built with **TensorFlow/Keras** and served through an interactive **Streamlit** interface, the app supports both image upload and real-time camera capture — with voice output for accessibility.

---

## ✨ Features

- **Banknote Classification** — Recognizes five Ethiopian Birr denominations: **5**, **10**, **50**, **100**, and **200** Birr
- **Dual Input Modes** — Upload an image file or capture directly from your device camera
- **Voice Output** — Automatically speaks the prediction result aloud using the Web Speech API for visually impaired users
- **Confidence Scoring** — Displays prediction confidence with a per-class probability breakdown
- **Transfer Learning** — Uses **MobileNetV2** pre-trained on ImageNet, fine-tuned on Ethiopian Birr banknote images for **94% test accuracy**

---

## 🏗️ Project Structure

```
├── app.py                  # Streamlit web application
├── models/
│   └── birr_mobilenetv2.keras   # Trained MobileNetV2 model (~11 MB)
├── notebook/
│   └── birr_banknote_cnn.ipynb  # Training notebook (EDA, CNN, Transfer Learning)
├── data/
│   └── Ethiopian_Currency/
│       ├── train/               # Training images (1,544 images across 11 classes)
│       └── test/                # Test images (375 images across 11 classes)
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md
```

---

## 📊 Dataset

The dataset contains **1,919 images** of Ethiopian Birr banknotes, organized into 11 classes (front and back of each denomination, plus background):

| Denomination | Train (front) | Train (back) | Test (front) | Test (back) |
|:------------|:-------------:|:------------:|:------------:|:-----------:|
| 5 Birr      | 148           | 148          | 36           | 36          |
| 10 Birr     | 148           | 148          | 36           | 36          |
| 50 Birr     | 148           | 148          | 36           | 36          |
| 100 Birr    | 148           | 148          | 36           | 36          |
| 200 Birr    | 148           | 148          | 36           | 36          |
| Background  | 64            | —            | 15           | —           |

The training pipeline groups front/back images into 5 denomination classes for final classification.

---

## 🧠 Model Architecture

Two models were developed and compared in the training notebook:

### 1. Custom CNN (Baseline)
- Built from scratch with convolutional and dense layers
- **Test Accuracy: 64%**
- Showed significant overfitting (high train accuracy, low validation accuracy)

### 2. MobileNetV2 Transfer Learning (Final Model) ✅
- Pre-trained **MobileNetV2** backbone (ImageNet weights, frozen)
- Custom classification head with Global Average Pooling and Dense layers
- Data augmentation (rotation, zoom, flip, shift)
- Callbacks: EarlyStopping + ReduceLROnPlateau
- **Test Accuracy: 94%**

| Class    | Precision | Recall | F1-Score |
|:---------|:---------:|:------:|:--------:|
| 5 Birr   | 0.94      | 0.90   | 0.92     |
| 10 Birr  | 0.96      | 0.97   | 0.97     |
| 50 Birr  | 0.96      | 0.97   | 0.97     |
| 100 Birr | 0.91      | 0.93   | 0.92     |
| 200 Birr | 0.94      | 0.93   | 0.94     |
| **Overall** | **0.94** | **0.94** | **0.94** |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/seelneas/ethiopian_birr_denomination_recognition.git
   cd ethiopian_birr_denomination_recognition
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # macOS/Linux
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 Usage

1. **Choose an input method** — "Upload Image" or "Use Camera"
2. **Provide a banknote image** — Upload a `.jpg`/`.jpeg`/`.png` file, or snap a photo using your device camera
3. **Click "🔍 Recognize Banknote"** — The model analyzes the image and predicts the denomination
4. **View results** — See the predicted denomination, confidence score, per-class probabilities, and hear the result spoken aloud

---

## 🛠️ Tech Stack

| Component       | Technology                         |
|:----------------|:-----------------------------------|
| Deep Learning   | TensorFlow / Keras                 |
| Model           | MobileNetV2 (Transfer Learning)    |
| Web Framework   | Streamlit                          |
| Image Processing| Pillow, NumPy                      |
| Voice Output    | Web Speech API (browser-native)    |
| Data Analysis   | Pandas, Matplotlib, Seaborn, Scikit-learn |

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---
