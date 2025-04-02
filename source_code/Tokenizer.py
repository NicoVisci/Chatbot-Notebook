import nltk


class TokenizerModule:
    """
    A specialized tokenizer for processing text with intent-specific filtering.

    This module uses NLTK to tokenize and tag input text, then filters tokens
    based on the identified intent and predefined rules. It includes specialized
    processing for product detection, ID detection, and medical term exclusion.
    """

    def __init__(self):
        """
        Initialize the TokenizerModule with necessary NLTK resources and filtering rules.

        Downloads required NLTK resources and sets up filtering lists:
        - allowed_tags_for_product_detection: POS tags relevant for product identification
        - allowed_tags_for_id_detection: POS tags relevant for ID information
        - excluded_words: List of medical and health-related terms to be filtered out
        """
        nltk.download([
            "names",
            "stopwords",
            "state_union",
            "twitter_samples",
            "movie_reviews",
            "averaged_perceptron_tagger_eng",
            "vader_lexicon",
            "punkt",
            "punkt_tab",
            "maxent_ne_chunker_tab",
            "words"
        ])
        # Tags for product detection - nouns and adjectives
        self.allowed_tags_for_product_detection = ['NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS']

        # Tags for ID detection - cardinal numbers and coordinating conjunctions
        self.allowed_tags_for_id_detection = ['CD', 'CC']

        # List of words to exclude from processing to avoid medical misclassification
        self.excluded_words = [
            # Illness / symptom words
            "sick", "fever", "cold", "ache", "cough", "chill", "pain", "flu", "rash",
            "sneeze", "dizzy", "weak", "cramp", "sniffle", "swell", "fatigue",

            # Vague or misleading medical-sounding words
            "remedy", "cure", "doctor", "dose", "therapy", "elixir", "tonic", "med",
            "clinic", "prescrip", "pharma",

            # Body-part related names
            "head", "heart", "skin", "muscle", "bone", "blood", "vein", "joint",

            # Medical-action words used in lifestyle branding
            "inject", "infuse", "heal", "boost", "revive", "restore",

            # Scientific/clean adjectives often misused
            "nano", "bio", "gen", "vital", "pure", "clean", "bright", "fresh",
            "clear", "active", "smart",

            # Purchase-related temporal terms
            "yesterday", "today", "tomorrow", "morning", "afternoon", "evening", "night",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "week", "month", "year", "date", "time", "day", "hour", "minute", "second",
            "weekend", "weekday", "recently", "soon", "later",

            # Purchase locations and methods
            "store", "shop", "mall", "market", "supermarket", "outlet", "online", "website",
            "app", "application", "site", "platform", "marketplace", "warehouse", "retailer",
            "seller", "vendor", "merchant", "dealer", "distributor", "supplier",

            # Purchase actions and processes
            "buy", "bought", "purchase", "purchased", "order", "ordered", "add", "added",
            "cart", "basket", "bag", "checkout", "check", "checked", "pay", "paid", "payment",
            "transaction", "receipt", "invoice", "bill", "sale", "sold", "selling", "sell",

            # Purchase history and status terms
            "history", "record", "log", "tracking", "track", "status", "pending", "processing",
            "shipped", "shipping", "delivered", "delivery", "return", "returned", "exchange",
            "refund", "refunded", "cancel", "canceled", "cancelled",

            # Shopping experience terms
            "experience", "service", "customer", "support", "help", "assistance", "issue",
            "problem", "question", "inquiry", "request", "feedback", "review", "rating",
            "satisfaction", "complaint", "concern"
        ]

    def tokenize(self, msg, intent):
        """
        Tokenize and filter text based on the provided intent.

        This method performs part-of-speech tagging on the input message and
        filters the tokens according to the specified intent and predefined rules.

        Args:
            msg (str): The input text message to tokenize
            intent (str): The identified intent, which determines filtering rules
                          Supported intents: 'self-identifying', 'purchase', 'preferences'

        Returns:
            list: A filtered list of tokens based on the intent and filtering rules
                 - For 'self-identifying': Returns only ID-related tokens
                 - For 'purchase'/'preferences': Returns only product-related tokens
                 - For other intents: Returns all tokens with their POS tags
        """
        tokens = nltk.pos_tag(nltk.word_tokenize(msg))
        if intent == 'self-identifying':
            return [token for token, tag in tokens if
                    tag in self.allowed_tags_for_id_detection and token not in self.excluded_words]
        if intent in ['purchase', 'preferences']:
            return [token for token, tag in tokens if
                    tag in self.allowed_tags_for_product_detection and token not in self.excluded_words]
        return [token for token, tag in tokens]