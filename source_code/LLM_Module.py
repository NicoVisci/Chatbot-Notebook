import yaml
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq


class LLM_Module:

    def __init__(self,
                 model_name = 'llama3-70b-8192',
                 context = 'You are a retail sales assistant for pharmaceutical products. Respond politely.'
        ):

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

        self.llm = Groq(model=model_name, api_key=apy_key)

        self.context = context


    def response(self, intent, msg):
        intentContext = ''
        if intent == 'Greeting':
            intentContext = 'Greet the user, politely clarify your role and invite the user to interact with you. Remember him that he should identify himself.'
        if intent == 'Self-identifying':
            intentContext = 'The user is identifying themselves. Acknowledge this information politely and ask how you can assist them with pharmaceutical products.'
        if intent == 'Purchase':
            intentContext = 'The user specified a recently purchased product. Acknowledge this information and update the user contextually to the response received.'
        if intent == 'Unrelated':
            intentContext = 'The user topic is beyond your scope, politely clarify that you are focused on pharmaceutical assistance.'
        if intent == 'Uncovered':
            intentContext = 'The user is referring to a task that is in the medical field but beyond your scope. Politely clarify that you are focused on pharmaceutical recommendation.'
        if intent == 'Recommendation':
            intentContext = 'The user is asking for pharmaceutical product recommendations. Explain the recommendations given, but remind them to consult with a medical professional before making a decision.'
        if intent == 'Preference':
            intentContext = 'The user is expressing preferences about pharmaceutical products. Acknowledge their preferences.'


        messages = [
            ChatMessage( role = "system", content = self.context + intentContext ),
            ChatMessage( role = "user", content = msg )
        ]

        return self.llm.chat(messages)
