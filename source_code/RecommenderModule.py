import pandas as pd
import requests
import csv
import tempfile
import time
from typing import Optional, Dict, Any


class RecommenderModule:
    """
    Module to handle interaction with the recommendation API.
    Provides methods to generate recommendations based on user input.
    """

    def __init__(self, api_url="http://localhost:8000", product_file = "data/PRODUCT_clean.csv", default_model_name:str = "Caser"):
        """
        Initialize the RecommenderModule with API configuration.

        Args:
            api_url (str): Base URL for the recommendation API
        """
        self.api_url = api_url
        self.model_name = default_model_name # Default model name, can be configured
        self.product_file = product_file

    def set_model(self, model_name):
        """
        Set the model to use for recommendations.

        Args:
            model_name (str): Name of the model to use
        """
        self.model_name = model_name

    def get_recommendations(self, purchases_file, user_id, k=10) -> Dict[str, Any]:
        """
        Get recommendations based on tokens.

        Args:
            purchases_file (str): Recent purchases file path.
            user_id (str, optional): User token for personalization.
            k (int, optional): Number of recommendations to return. Defaults to 10.

        Returns:
            dict: Recommendations from the API or error message
        """
        # Generate user token if not provided
        if user_id is None:
            return {"status": "error", "message": "User not identified. Please identify before requesting recommendations."}

        try:
            # Make prediction request
            task_id = self._request_prediction(purchases_file, user_id, k)

            # Wait for and retrieve results
            recommendations = self._wait_for_task_result(task_id)

            return {
                "status": "success",
                "recommendations": recommendations
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _request_prediction(self, tokens_file, user_token, k):
        """
        Send prediction request to API.

        Args:
            tokens_file (str): Path to CSV file with tokens
            user_token (str): User token for personalization
            k (int): Number of recommendations to return

        Returns:
            str: Task ID for the prediction job
        """
        files = {'file': open(tokens_file, 'rb')}
        data = {
            'user_token': user_token,
            'k': k,
            'model': self.model_name
        }

        # Send prediction request
        response = requests.post(f'{self.api_url}/predict', files=files, data=data)

        if response.status_code != 202:  # 202 Accepted
            raise Exception(f"API request failed with status {response.status_code}: {response.text}")

        return response.json()['task_id']

    def _wait_for_task_result(self, task_id, max_attempts=50, initial_delay=1, max_delay = 500, backoff_factor=1.5):
        """
        Wait for and retrieve task result with exponential backoff.

        Args:
            task_id (str): Task ID to check
            max_attempts (int, optional): Maximum number of polling attempts. Defaults to 30.
            initial_delay (int, optional): Initial delay between attempts in seconds. Defaults to 1.
            backoff_factor (float, optional): Factor to increase delay by each attempt. Defaults to 1.5.

        Returns:
            Any: Task result from the API

        Raises:
            Exception: If task fails or times out
        """
        delay = initial_delay

        for attempt in range(max_attempts):
            # Check task status
            status_response = requests.get(f'{self.api_url}/task-status/{task_id}')

            if status_response.status_code != 202:
                raise Exception(f"Status check failed with code {status_response.status_code}")

            status = status_response.json()['status']

            if status == 'SUCCESS':
                # Retrieve task result
                result_response = requests.get(f'{self.api_url}/task-result/{task_id}')

                if result_response.status_code != 200:
                    raise Exception(f"Result retrieval failed with code {result_response.status_code}")

                return result_response.json().get('result')

            elif status == 'FAILURE':
                raise Exception("Task processing failed on the server")

            # Wait before next attempt with exponential backoff
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)  # Cap at 30 seconds

        raise Exception(f"Task result retrieval timed out after {max_attempts} attempts")

    def create_product_sample_csv(self, sample_size=10):
        """
        Create a temporary CSV file with a sample of products from the product catalog.

        Args:
            sample_size (int): Number of products to sample

        Returns:
            str: Path to the created temporary file
        """
        try:
            # Read the product catalog
            products_df = pd.read_csv(self.product_file)

            # Check if we have enough products
            if len(products_df) < sample_size:
                sample_size = len(products_df)

            # Sample products randomly
            sampled_products = products_df.sample(n=sample_size)

            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                # Write sampled products to CSV
                sampled_products.to_csv(temp_file, index=False)

            return temp_file.name

        except Exception as e:
            # If there's an error with the product file, fallback to a simple tokens file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                writer = csv.writer(temp_file)
                # Write header
                writer.writerow(['product_id'])
                # Write sample product IDs
                for i in range(sample_size):
                    writer.writerow([f"product_{i + 1}"])

            # Log the error
            print(f"Error reading product file: {str(e)}. Using fallback.")
            return temp_file.name