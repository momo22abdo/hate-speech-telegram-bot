# Utility module for generating explainability using LIME

import logging
import traceback
from lime.lime_text import LimeTextExplainer
from utils.classification import classifier_fn

# Set up logging
logger = logging.getLogger(__name__)


def explain_prediction(text: str, prediction: dict) -> str:
    """
    Generate explanation for the prediction using LIME.

    Args:
        text (str): The input text.
        prediction (dict): The classification result.

    Returns:
        str: Explanation string with important words and scores.
    """
    logger.info("Generating explanation for text: %s, prediction: %s", text, prediction)

    # Check if text is too short
    word_count = len(text.split())
    if word_count <= 3:
        logger.warning("Text is too short for meaningful explanation: %s (%d words)", text, word_count)
        return "Text is too short to generate a meaningful explanation (minimum 4 words required for better results)."

    try:
        class_names = ['Hate Speech', 'Offensive', 'Normal']
        # Use fewer features for short texts
        num_features = min(word_count, 5)  # Limit to 5 or number of words
        explainer = LimeTextExplainer(class_names=class_names)
        exp = explainer.explain_instance(
            text,
            classifier_fn,
            num_features=num_features,
            num_samples=500,  # Reduced for faster processing
            top_labels=3  # Explain all 3 classes to ensure the predicted label is included
        )

        # Get the index of the predicted label
        label_idx = class_names.index(prediction['label'])

        # Check if the label is in local_exp
        if label_idx not in exp.local_exp:
            logger.warning("Predicted label %d not in local_exp keys: %s", label_idx, list(exp.local_exp.keys()))
            return "Explanation not available for the predicted label (try a longer text)."

        # Get important words for the predicted class
        important_words = exp.as_list(label=label_idx)

        if not important_words:
            logger.warning("No important words found for text: %s", text)
            return "No significant words found to explain the decision."

        explanation = "Important words influencing the decision:\n"
        for word, score in important_words:
            explanation += f"{word}: {score:.4f}\n"

        logger.info("Explanation generated successfully: %s", explanation)
        return explanation
    except Exception as e:
        # Log the full stack trace for debugging
        logger.error("Error in explain_prediction: %s\n%s", str(e), traceback.format_exc())
        return f"Failed to generate explanation: {str(e)}"