# Utility module for text preprocessing and classification - enhanced version

import joblib
import os
import numpy as np
import re
import string
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Class labels (aligned with Measuring Hate Speech dataset: 0=Hate Speech, 1=Offensive, 2=Normal)
LABELS = {0: 'Hate Speech', 1: 'Offensive', 2: 'Normal'}


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
        r'\bcan\'?t\s+(?:take|live|go\s+on)\b.*(?:anymore|any\s*more)\b',
        r'\bkill\s+my\s*self\b',
        r'\bhurt\s+my\s*self\b'
    ]

    return any(re.search(pattern, text_lower) for pattern in self_harm_patterns)


def detect_threat_patterns(text):
    """Detect threat patterns directed at others."""
    text_lower = text.lower()

    # Threat patterns directed at others
    threat_patterns = [
        r'\b(?:kill|murder|shoot|stab|beat)\s+(?:you|them|him|her)\b',
        r'\bi\s+(?:will|gonna)\s+(?:kill|murder|hurt)\s+you\b',
        r'\byou\s+(?:should|will|gonna)\s+die\b',
        r'\bhope\s+you\s+die\b'
    ]

    return any(re.search(pattern, text_lower) for pattern in threat_patterns)


# Paths to pre-trained model and vectorizer
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'classifier.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'vectorizer.joblib')

# Global variables to hold the model and vectorizer
classifier = None
vectorizer = None


def load_models():
    """Load the model and vectorizer lazily."""
    global classifier, vectorizer

    if classifier is None or vectorizer is None:
        try:
            classifier = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            logger.info("Model and vectorizer loaded successfully")
        except FileNotFoundError as e:
            logger.error("Model files not found. Please run train_model.py first: %s", str(e))
            raise
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise

    return classifier, vectorizer


def preprocess_text(text: str) -> str:
    """
    Preprocess the input text: lowercase, remove punctuation and emojis.

    Args:
        text (str): The input text.

    Returns:
        str: Preprocessed text. Returns a space if input is empty or invalid.
    """
    if not text or not isinstance(text, str):
        logger.debug("Invalid text input: %s", text)
        return " "
    text = text.lower()
    # Keep important punctuation that affects meaning
    text = re.sub(r'[^\w\s!?.,-]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        logger.debug("Text is empty after preprocessing")
        return " "
    logger.debug("Preprocessed text: %s", text)
    return text


def classify_text(text: str) -> dict:
    """
    Classify the input text using the pre-trained model with enhanced handling.

    Args:
        text (str): The text to classify.

    Returns:
        dict: {'label': str, 'confidence': float, 'flags': list}
    """
    logger.info("Classifying text: %s", text)

    # Load models if not already loaded
    clf, vec = load_models()

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Handle empty text after preprocessing
    if processed_text == " ":
        logger.warning("Text is empty after preprocessing")
        return {'label': 'Normal', 'confidence': 100.0, 'flags': []}

    # Check for special patterns
    flags = []

    # Check for self-harm patterns
    if detect_self_harm_patterns(text):
        flags.append('SELF_HARM')
        logger.warning("⚠️ Self-harm content detected: %s", text)

    # Check for threat patterns
    if detect_threat_patterns(text):
        flags.append('THREAT')
        logger.warning("⚠️ Threat content detected: %s", text)

    # Vectorize the text
    vectorized_text = vec.transform([processed_text])

    # Predict probabilities
    probs = clf.predict_proba(vectorized_text)[0]

    # Get predicted class and confidence (max probability)
    pred_class = np.argmax(probs)
    confidence = np.max(probs)

    # Adjust classification based on special patterns
    original_label = LABELS.get(pred_class, 'Unknown')
    final_label = original_label

    # Special handling for self-harm content
    if 'SELF_HARM' in flags:
        # Self-harm content should be treated as concerning but differently from hate speech toward others
        final_label = f"{original_label} (Self-harm detected)"
        logger.info("Applied self-harm flag to classification")

    # Special handling for threats
    if 'THREAT' in flags and pred_class != 0:  # If not already classified as hate speech
        final_label = "Hate Speech (Threat detected)"
        confidence = max(confidence, 0.85)  # Boost confidence for clear threats
        logger.info("Applied threat flag to classification")

    result = {
        'label': final_label,
        'confidence': round(confidence * 100, 2),
        'flags': flags
    }
    logger.info("Final classification result: %s", result)
    return result


def classifier_fn(texts: list) -> np.ndarray:
    """
    Classifier function for LIME: takes a list of texts, preprocesses, vectorizes, and returns probabilities.

    Args:
        texts (list): List of texts.

    Returns:
        np.ndarray: Probability arrays.
    """
    logger.debug("Processing %d texts for LIME: first few %s", len(texts), texts[:5])

    # Load models if not already loaded
    clf, vec = load_models()

    processed = [preprocess_text(t) if t else " " for t in texts]
    # Replace empty strings with a space to avoid vectorizer issues
    processed = [" " if not p else p for p in processed]
    try:
        vecs = vec.transform(processed)
        probs = clf.predict_proba(vecs)
        logger.debug("Generated probabilities for LIME: shape %s, first %s", probs.shape, probs[0])
        return probs
    except Exception as e:
        logger.error("Error in classifier_fn: %s", str(e))
        raise