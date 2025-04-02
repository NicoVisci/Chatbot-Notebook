import os
import pandas as pd
from typing import Optional, Dict, Any

from thefuzz import process


class UserDataManagementModule:
    """
    A class that manages user data, including identity and purchase history.
    It updates user information based on detected intents and stores data in files.
    """

    def __init__(self, data_directory: str = "data/user_data", products_file: str = "data/PRODUCT_clean.csv"):
        """
        Initialize the user data management module.

        Args:
            data_directory: Directory where user data files will be stored
            products_file: Path to the CSV file containing product information
        """
        self.user_id = None
        self.data_directory = data_directory
        self.products_file = products_file

        # Load products dataframe
        try:
            self.products_df = pd.read_csv(self.products_file)
            self.products_dict = {desc: idx for idx, desc in enumerate(self.products_df['D_PRODUCT'])}
        except Exception as e:
            print(f"Error loading products file: {str(e)}")
            self.products_df = pd.DataFrame()
            self.products_dict = {}

        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)

    def process_intent(self, intent: str, tokens: str) -> Dict[str, Any]:
        """
        Process the detected intent and update user data accordingly.

        Args:
            intent: The detected intent from the intent recognition module
            tokens: The tokens extracted from the original user message

        Returns:
            A dictionary with the operation result
        """
        if intent == "Self-identifying":
            return self._handle_self_identifying(tokens)
        elif intent == "Purchase":
            return self._handle_purchase(tokens)
        else:
            return {"status": "no_action", "message": f"No action taken for intent: {intent}"}

    def _handle_self_identifying(self, tokens: str) -> Dict[str, Any]:
        """
        Handle self-identifying intent by extracting and setting user ID.

        Args:
            tokens: The tokens containing self-identification

        Returns:
            A dictionary with the operation result
        """
        # Extraction of potential user ID
        potential_id = None
        for token in tokens.split():
            # First check if the string can be converted to an integer
            try:
                # Convert to integer
                num = int(token)

                # Check if it has exactly 8 digits by converting to string and checking length
                # Use abs() to handle negative numbers
                if len(str(abs(num))) == 8:
                    potential_id = str(token)
                    break
            except ValueError:
                pass
                # If conversion fails, it's not a valid integer

        # Clean up ID to make it filename-safe
        if potential_id:
            self.user_id = potential_id

            # Create user profile file if it doesn't exist
            # self._ensure_user_profile_exists()

            return {
                "status": "success",
                "message": f"User ID updated to {self.user_id}"
            }

        return {"status": "error", "message": "Could not extract user ID from message"}

    def _handle_purchase(self, product_token: str) -> Dict[str, Any]:
        """
        Handle purchase intent by recording purchase information.

        Args:
            product_token: The token containing the name of the purchased item

        Returns:
            A dictionary with the operation result
        """
        if not self.user_id:
            return {"status": "error", "message": "User not identified. Please identify before making purchases."}

        # Check if the product exists in the products dataframe
        product = self._match_product(product_token)

        if not product:
            return {
                "status": "error",
                "message": f"Product '{product_token}' does not exist in the product catalog."
            }

        try:
            # Save purchase to CSV
            self._save_purchase_to_csv(product)

            return {
                "status": "success",
                "message": f"Purchase '{product_token}' recorded for user {self.user_id}"
            }

        except Exception as e:
            return {"status": "error", "message": f"Failed to record purchase: {str(e)}"}

    def _match_product(self, product_token: str, threshold: int = 75) -> Optional[pd.DataFrame]:
        """
        Return the product selected from the catalog if it exists.

        Args:
            product_token: The name of the product to check

        Returns:
            The product entry if it exists, None otherwise
        """
        if self.products_df.empty:
            return None

        # Find matches using process.extractBests
        matches = process.extractBests(product_token, self.products_dict.keys(), score_cutoff=threshold, limit=1)

        # If no matches found above threshold
        if not matches:
            print(f"No matches found above threshold score of {threshold}")
            return None

        # Create a list of indices for the matches
        matched_indices = [self.products_dict[match[0]] for match in matches]

        # Create a results DataFrame with the matches
        results = self.products_df.iloc[matched_indices].copy()

        return results

    def _save_purchase_to_csv(self, purchase) -> None:
        """
        Save purchase details to a CSV file.

        Args:
            purchase_details: The purchase details to save
        """
        csv_file = os.path.join(self.data_directory, f"{self.user_id}_purchases.csv")

        # Create new dataframe with the purchase
        purchase_df = pd.DataFrame(purchase)

        # Append to existing CSV or create new one
        if os.path.exists(csv_file):
            # Read existing CSV and append new purchase
            existing_df = pd.read_csv(csv_file)
            updated_df = pd.concat([existing_df, purchase_df], ignore_index=True)
            updated_df.to_csv(csv_file, index=False)
        else:
            # Create new CSV with headers
            purchase_df.to_csv(csv_file, index=False)

    def _get_purchase_file_path(self) -> str:
        """
        Get the path to the user's purchase file.

        Returns:
            File path for the user purchases
        """
        return os.path.join(self.data_directory, f"{self.user_id}_purchases.csv")

    def get_user_id(self) -> Optional[str]:
        """
        Get the current user ID.

        Returns:
            The current user ID or None if not set
        """
        return self.user_id

    def get_user_purchases(self):
        """
        Get the list of purchases for the current user.

        Returns:
            List of purchase dictionaries or empty list if no purchases or user not identified
        """
        if not self.user_id:
            return []

        purchase_file = self._get_purchase_file_path()

        if os.path.exists(purchase_file):
            return pd.read_csv(purchase_file)
        else:
            return []