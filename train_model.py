# Script to train the hate speech detection model - simplified version

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import logging
import re

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_self_harm_patterns(text):
    """Detect self-harm language patterns."""
    text_lower = text.lower()

    # Self-harm indicators
    self_harm_patterns = [
        r'\bi\s+(?:will|gonna|want\s+to|wanna)\s+(?:kill|hurt|harm)\s+(?:myself|my\s*self)\b',
        r'\bi\s+(?:will|gonna|want\s+to|wanna)\s+(?:suicide|die)\b',
        r'\bsuicidal\b',
        r'\bkilling?\s+myself\b',
        r'\bend\s+(?:my|it)\s+(?:all|life)\b',
        r'\bcan\'?t\s+(?:take|live|go\s+on)\b.*(?:anymore|any\s*more)\b'
    ]

    return any(re.search(pattern, text_lower) for pattern in self_harm_patterns)


def preprocess_text(text):
    """Enhanced text preprocessing."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Keep important punctuation that affects meaning
    text = re.sub(r'[^\w\s!?.,-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_prepare_data():
    """Load and prepare the dataset with corrected mapping."""
    logger.info("Loading dataset...")
    df = pd.read_csv(r"C:\Users\Momo\PycharmProjects\hate_speech_bot\data\hate_speech_data.csv")
    logger.info(f"Loaded {len(df)} samples.")

    # Apply preprocessing
    df['text'] = df['text'].apply(preprocess_text)

    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    logger.info(f"After cleaning: {len(df)} samples.")

    def map_to_label(score):
        """
        Enhanced mapping with better boundary definitions:
        - High toxicity scores (≥ 1.5) = Hate Speech (severe slurs, threats, extreme toxicity)
        - Medium toxicity scores (0.0 to 1.5) = Offensive (insults, mild profanity, rude language)
        - Low/negative toxicity scores (< 0.0) = Normal (neutral, positive, supportive content)
        """
        if score >= 1.5:
            return 0  # Hate Speech (only the most toxic content)
        elif score >= 0.0:
            return 1  # Offensive (mildly problematic content)
        else:
            return 2  # Normal (neutral to positive content)

    df['label'] = df['sentiment'].apply(map_to_label)

    # Print distribution for verification
    logger.info("Label distribution:")
    logger.info(df['label'].value_counts().sort_index())

    # Print sample data for manual verification
    logger.info("\nSample data for verification:")
    for label in [0, 1, 2]:
        logger.info(f"\n--- Label {label} samples ---")
        samples = df[df['label'] == label].head(3)
        for _, row in samples.iterrows():
            logger.info(f"Text: '{row['text'][:80]}...' | Sentiment: {row['sentiment']:.2f}")

    return df['text'].values, df['label'].values


def create_classifier():
    """Create a standard classifier without custom wrapper."""
    # TF-IDF with enhanced parameters
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,  # Use sublinear scaling
        analyzer='word'
    )

    # Enhanced SVM with better parameters
    base_svm = LinearSVC(
        max_iter=3000,
        C=0.8,  # Slightly lower C for better generalization
        class_weight='balanced',
        random_state=42,
        dual=False  # More efficient for large feature sets
    )

    # Calibrate for probability estimates
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)

    return tfidf, calibrated_svm


def train_model():
    """Train the SVM model with standard sklearn components only."""
    os.makedirs('models', exist_ok=True)
    texts, labels = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    logger.info("Creating classifier...")
    vectorizer, classifier = create_classifier()

    logger.info("Vectorizing texts...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logger.info("Training classifier...")
    classifier.fit(X_train_vec, y_train)

    logger.info("Evaluating model...")
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    logger.info(f"Model evaluation metrics:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Detailed classification report
    logger.info("\nDetailed Classification Report:")
    class_names = ['Hate Speech', 'Offensive', 'Normal']
    logger.info(classification_report(y_test, y_pred, target_names=class_names))

    # Save models
    joblib.dump(classifier, 'models/classifier.joblib')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    logger.info("Model and vectorizer saved successfully")

    # Test examples
    test_examples = [
        # Original examples
        "i hate u",
        "i will kill my self",
        "You're an idiot",
        "You're an idiot and I hate you very much",
        "This is offensive!",
        "I love u",

        # Self-harm examples
        "I want to kill myself",
        "I'm going to end it all",
        "I can't take it anymore, I want to die",

        # Boundary testing examples
        "You're stupid",  # Mild insult
        "Fuck you, you worthless piece of shit",  # Strong profanity
        "I hope you die in a fire, you fucking retard",  # Hate speech
        "That's really annoying",  # Normal criticism
        "You did great today!",  # Positive
    ]

    logger.info("\nTesting examples:")
    for text in test_examples:
        processed = preprocess_text(text)
        vec = vectorizer.transform([processed])
        pred = classifier.predict(vec)[0]
        prob = classifier.predict_proba(vec)[0]
        max_prob = prob.max()

        # Check for self-harm
        is_self_harm = detect_self_harm_patterns(text)
        self_harm_flag = " [SELF-HARM DETECTED]" if is_self_harm else ""

        label_map = {0: 'Hate Speech', 1: 'Offensive', 2: 'Normal'}
        logger.info(f"'{text}' → {label_map[pred]} ({max_prob:.2%}){self_harm_flag}")


if __name__ == '__main__':
    train_model()