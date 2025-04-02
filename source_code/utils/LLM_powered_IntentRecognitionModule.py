import yaml
from llama_index.llms.groq import Groq


class LLM_powered_IntentRecognitionModule:
    """
    A class that recognizes user intents by using the Groq API to query the llama3-70b-8192 model.
    """

    def __init__(self):
        """
        Initialize the intent recognition module.
        """

        api_config_path = 'LLM_ApiKey.yml'
        try:
            with open(api_config_path, 'r') as config_file:
                apy_key = yaml.safe_load(config_file).get('key')
        except FileNotFoundError:
            print(f"LLM Api file not found: {api_config_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

        if apy_key is None:
            raise ValueError("LLM Api key not found.")

        self.model_name = "llama-3.3-70b-versatile" # llama3-70b-8192
        self.llm = Groq(model=self.model_name, api_key=apy_key)

        self.valid_intents = [
            "Self-identifying",
            "Purchase",
            "Preferences",
            "Recommendation request",
            "Uncovered",
            "Unrelated"
        ]

    def recognize_intent(self, user_message: str) -> str:
        """
        Recognizes the intent behind a user message.

        Args:
            user_message: The message from the user to analyze

        Returns:
            A string representing the recognized intent
        """
        prompt = (f"The user has written the following message: '{user_message}'. "
                  f"Determine which of the following intents is the most appropriate: "
                  f"[User Self-identifying, Purchase, Preferences, Recommendation request, Uncovered, Unrelated]. "
                  f"Return only the name of the most suitable intent.")

        payload = {
            "model": self.llm,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Use low temperature for more deterministic results
            "max_tokens": 20  # The response should be very short
        }

        try:
            intent = self.llm.complete(prompt)
            # Clean up response to ensure it matches one of our valid intents
            for valid_intent in self.valid_intents:
                if valid_intent in intent:
                    return valid_intent

            # If no match found, return "Unrelated" as default
            return "Unrelated"

        except Exception as e:
            print(f"Error calling Groq API: {e}")
            return "Unrelated"  # Default fallback

    def __call__(self, user_message: str) -> str:
        """
        Makes the class callable, allowing it to be used like a function.

        Args:
            user_message: The message from the user to analyze

        Returns:
            A string representing the recognized intent
        """
        return self.recognize_intent(user_message)