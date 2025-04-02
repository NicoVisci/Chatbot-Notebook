
from transformers import pipeline

class SentimentAnalysisModule:

    def __init__(self):
        # self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

    def analyze(self, msg):
        sentiment_map = {
            'POS': 'Positive',
            'NEU': 'Neutral',
            'NEG': 'Negative',
        }
        sentiment_data = self.sentiment_pipeline(msg)[0]

        return sentiment_map[sentiment_data['label']]