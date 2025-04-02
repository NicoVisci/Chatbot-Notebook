from source_code.IntentModule import IntentModule


class IntentRecognitionModule:
    """
    A module for recognizing and classifying user intents in text input.

    This class implements a hierarchical intent recognition system with three levels:
    1. Initial phase - Determines if text is unrelated, medical, or an update request
    2. Update recognition - Further classifies update requests as self-identifying or purchase
    3. Medical recognition - Further classifies medical queries as recommendations, preferences, or uncovered services

    The module uses specialized IntentModule instances for each classification level.
    """

    def __init__(self):
        """
        Initialize the IntentRecognitionModule with specialized intent classifiers.

        Creates three IntentModule instances for the hierarchical intent recognition:
        - InitialPhaseModule: First-level classification
        - UpdateRecognitionModule: Second-level classification for update intents
        - MedicalRecognitionModule: Second-level classification for medical intents
        """
        self.InitialPhaseModule = IntentModule('Initial_Intent', ["Unrelated", "Medical", "Update"])
        self.UpdateRecognitionModule = IntentModule('Update_Intent', ["Self-identifying", "Purchase"])
        self.MedicalRecognitionModule = IntentModule('Medical_Intent', ["Recommendation", "Preferences", "Uncovered"])

    def recognize_intent(self, text):
        """
        Recognize the intent of the given text using the hierarchical classification system.

        Args:
            text (str): The input text to analyze for intent recognition

        Returns:
            str: The recognized intent, which may be one of:
                - "Unrelated"
                - "Medical" -> ["Recommendation", "Preferences", "Uncovered"]
                - "Update" -> ["Self-identifying", "Purchase"]
        """
        intent = self.InitialPhaseModule.predict(text)
        if intent == 'Update':
            intent = self.UpdateRecognitionModule.predict(text)
        if intent == 'Medical':
            intent = self.MedicalRecognitionModule.predict(text)
        return intent


