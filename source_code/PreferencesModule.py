import os
from typing import Optional, Dict, Any

import pandas as pd
from rapidfuzz import fuzz, process
from thefuzz import fuzz, process


class PreferencesModule:
    """
    A module for managing and tracking user preferences related to products.

    This class handles loading product data from files, matching product descriptions
    using fuzzy matching, and tracking user sentiment towards products over time.
    Each user's preferences are stored in a separate file.

    Attributes:
        product_df (pandas.DataFrame): DataFrame containing product data with K_PRODUCT and D_PRODUCT columns
        product_dict (dict): Dictionary mapping product descriptions to their indices
        preference_dfs (dict): Dictionary mapping user IDs to their preference DataFrames
        preference_file_path (str): Base path for preference files without user ID
        preferences_dir (str): Directory where preference files are stored
    """

    def __init__(self,
                 product_input_file: str = "data/PRODUCT_clean.csv",
                 product_output_file: str = "data/PRODUCT_short.csv",
                 preferences_dir: str = "data/preferences",
                 preference_file_path: str = "_preferences.csv"
                 ):
        """
        Initialize the PreferencesModule with product and preference data.

        Loads product data from either an existing output file or processes it from
        the input file. Also prepares the structure for managing user-specific preferences.

        Parameters:
            product_input_file (str): Path to the full product data CSV file
            product_output_file (str): Path to save/load the simplified product data
            preferences_dir (str): Directory to save/load user preferences
            preference_file_path (str): Base path for preference files
        """
        if os.path.exists(product_output_file):
            # If it exists, load directly from the output file
            self.product_df = pd.read_csv(product_output_file)
        else:
            # If it doesn't exist, process from the input file
            df = pd.read_csv(product_input_file)
            # Select only the K_PRODUCT and D_PRODUCT columns
            self.product_df = df[['K_PRODUCT', 'D_PRODUCT']]
            # Save to a new CSV file
            self.product_df.to_csv(product_output_file, index=False)
        self.product_dict = {desc: idx for idx, desc in enumerate(self.product_df['D_PRODUCT'])}

        # Set up preferences directory
        self.preferences_dir = preferences_dir
        if not os.path.exists(preferences_dir):
            os.makedirs(preferences_dir)

        # Initialize dictionary to store user preference DataFrames
        self.preference_dfs = {}

    def _get_user_preference_path(self, user_id: str) -> str:
        """
        Generate the file path for a specific user's preferences.

        Parameters:
            user_id (str): The ID of the user

        Returns:
            str: Path to the user's preference CSV file
        """
        return os.path.join(self.preferences_dir, f"{user_id}_preferences.csv")

    def _load_user_preferences(self, user_id: str) -> pd.DataFrame:
        """
        Load preferences for a specific user. Creates a new preferences DataFrame if none exists.

        Parameters:
            user_id (str): The ID of the user

        Returns:
            pandas.DataFrame: The user's preference DataFrame
        """
        preference_file = self._get_user_preference_path(user_id)

        if os.path.exists(preference_file):
            # If a preference file exists for this user, load it
            return pd.read_csv(preference_file)
        else:
            # Create a new DataFrame for this user
            return pd.DataFrame(columns=['K_PRODUCT', 'SENTIMENT', 'SENTIMENT_SCORE', 'INTERACTIONS'])

    def update_preferences(self, user_id: str, product_tokens: str, sentiment: str) -> Dict[str, Any]:
        """
        Update preferences for a specific user based on products that match the given tokens.

        Uses fuzzy matching to find products that match the provided tokens,
        then updates sentiment information for each matched product for the specified user.

        Parameters:
            user_id (str): The ID of the user whose preferences are being updated
            product_tokens (str): Text to match against product descriptions
            sentiment (str): The sentiment to record ('Positive', 'Neutral', or 'Negative')

        Returns:
            int: Number of products affected by the update
        """
        if not user_id:
            return {"status": "error", "message": "User not identified. Please identify before expressing preferences."}

        # Load user preferences if not already loaded
        if user_id not in self.preference_dfs:
            self.preference_dfs[user_id] = self._load_user_preferences(user_id)

        # Find matching products
        affected_products = self._match(product_tokens)

        # Update sentiment for each product
        for product in affected_products:
            self._update_product_sentiment(user_id, product, sentiment)

        # Save user preferences to their dedicated file
        user_preference_file = self._get_user_preference_path(user_id)
        self.preference_dfs[user_id].to_csv(user_preference_file, index=False)

        return {"status": "success", "message": f"{len(affected_products)} products affected by an update based on user preferences"}

    def _match(self, tokens, threshold: int = 75):
        """
        Find products that match the given tokens using fuzzy matching.

        Parameters:
            tokens (str): Text to match against product descriptions
            threshold (int): Minimum similarity score (0-100) required for a match

        Returns:
            list: List of K_PRODUCT values for matched products
        """
        # Find matches using process.extractBests
        matches = process.extractBests(tokens, self.product_dict.keys(), score_cutoff=threshold, limit=None)

        # If no matches found above threshold
        if not matches:
            print(f"No matches found above threshold score of {threshold}")
            return []

        # Create a list of indices for the matches
        matched_indices = [self.product_dict[match[0]] for match in matches]

        # Create a results DataFrame with the matches
        results = self.product_df.iloc[matched_indices].copy()

        # Add the match score to the results
        results['MATCH_SCORE'] = [match[1] for match in matches]

        # Sort by match score in descending order
        results = results.sort_values('MATCH_SCORE', ascending=False)

        return results['K_PRODUCT'].tolist()

    def _update_product_sentiment(self, user_id: str, product, sentiment: str, method: str = 'weighted'):
        """
        Update the sentiment for a specific product in a user's preference DataFrame.

        Parameters:
            user_id (str): The ID of the user whose preferences are being updated
            product (int): The product ID to update
            sentiment (str): The sentiment expressed by the user ('Positive', 'Neutral', 'Negative')
            method (str, optional): Method for calculating sentiment:
                - 'weighted': Gives more weight to recent sentiments
                - 'average': Simple average of all sentiments

        Returns:
            dict: Empty dictionary (return value appears unused)

        Raises:
            ValueError: If sentiment is not one of the valid options
        """
        # Validate sentiment input
        valid_sentiments = ['Positive', 'Neutral', 'Negative']
        if sentiment not in valid_sentiments:
            raise ValueError(f"Invalid sentiment. Must be one of {valid_sentiments}")

        # Convert sentiment to numeric values
        sentiment_map = {
            'Positive': 1,
            'Neutral': 0,
            'Negative': -1
        }

        # Get the user's preference DataFrame
        preference_df = self.preference_dfs[user_id]

        # Check if product exists in the DataFrame
        product_exists = preference_df[preference_df['K_PRODUCT'] == product]

        if len(product_exists) == 0:
            # Add new product if it doesn't exist
            new_row = pd.DataFrame({
                'K_PRODUCT': [product],
                'SENTIMENT': [sentiment],
                'SENTIMENT_SCORE': [sentiment_map[sentiment]],
                'INTERACTIONS': [1]
            })
            self.preference_dfs[user_id] = pd.concat([preference_df, new_row], ignore_index=True)
        else:
            # Update existing product
            idx = product_exists.index[0]

            if method == 'weighted':
                # Weighted calculation with decay
                current_score = preference_df.loc[idx, 'SENTIMENT_SCORE']
                current_interactions = preference_df.loc[idx, 'INTERACTIONS']

                # Weighted average with exponential decay
                new_score = (current_score * current_interactions + sentiment_map[sentiment]) / (
                        current_interactions + 1)

                preference_df.loc[idx, 'SENTIMENT_SCORE'] = new_score
                preference_df.loc[idx, 'INTERACTIONS'] += 1

                # Update overall sentiment based on the new score
                if new_score > 0.5:
                    preference_df.loc[idx, 'SENTIMENT'] = 'Positive'
                elif new_score < -0.5:
                    preference_df.loc[idx, 'SENTIMENT'] = 'Negative'
                else:
                    preference_df.loc[idx, 'SENTIMENT'] = 'Neutral'

            elif method == 'average':
                # Simple average method
                preference_df.loc[idx, 'SENTIMENT'] = sentiment

        return {}