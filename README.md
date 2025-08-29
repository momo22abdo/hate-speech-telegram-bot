# Hate Speech Telegram Bot 🤖

A Telegram bot that detects **Hate Speech**, **Offensive Language**, and **Self-harm patterns** in user messages.  
It uses an **SVM classifier with TF-IDF features** and includes explainability & user statistics.

---

## 🚀 Features
- Detects:
  - Hate Speech
  - Offensive language
  - Normal text
  - Self-harm patterns
- Explainability with word highlights
- User stats and charts
- Supports retraining with new datasets
- Logs for debugging

---

## 📂 Project Structure
hate_speech_bot/
│── main.py # Telegram bot main script
│── train_model.py # Train and save model
│── config.py # Store Telegram Bot Token
│── requirements.txt # Dependencies
│
├── data/
│ └── hate_speech_data.csv # Dataset
│
├── models/
│ ├── classifier.joblib
│ ├── vectorizer.joblib
│ └── label_map.joblib
│
├── utils/
│ ├── classification.py
│ ├── explain.py
│ └── stats.py
│
├── stats.db # User stats database
├── bot.log # Bot logs
└── confusion_matrix.png # Saved evaluation result

yaml
Copy code

---

## 🛠️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/momo22abdo/hate-speech-telegram-bot.git
cd hate-speech-telegram-bot

# 2. Create a virtual environment & install requirements
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 3. Add your Telegram Bot Token inside config.py
TELEGRAM_BOT_TOKEN = "your-token-here"
📊 Train Model
bash
Copy code
python train_model.py
This will:

Train with Stratified K-Fold cross-validation

Handle class imbalance (SMOTE)

Save classifier, vectorizer, and label map

Export confusion matrix (confusion_matrix.png)

🤖 Run the Bot
bash
Copy code
python main.py
📌 Notes
If you change the dataset, retrain the model with train_model.py.

Check logs in bot.log for debugging.

User stats are stored in stats.db.
