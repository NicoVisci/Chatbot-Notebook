import gc
import os

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfRetriever:
    def __init__(self, product_key='K_PRODUCT', product_description='D_PRODUCT', input_file='data/PRODUCT_short.csv', output_file='data/TF_IDF.csv'):
        """
        Initialize the TF-IDF based retrieval system.

        Parameters:
        -----------
        data : pandas.DataFrame, optional
            DataFrame containing product data
        product_key : str, default='K_PRODUCT'
            Column name for product identifier/key
        product_description : str, default='D_PRODUCT'
            Column name for product description text
        """
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.9,
            min_df=2
        )
        self.product_key = product_key
        self.product_description = product_description

        if os.path.exists(output_file):
            # If it exists, load directly from the output file
            self.tfidf_matrix = pd.read_csv(output_file)
        else :
            self.tfidf_matrix = None

        self.data = pd.read_csv(input_file)
        self.product_keys = self.data[self.product_key].values

        self.input_file = input_file
        self.output_file = output_file

    def fit(self):
        """
        Fit the TF-IDF vectorizer on the product descriptions without copying the data.

        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing product data (reference will be stored, not copied)
        """
        if self.tfidf_matrix is not None:
            return {}

        # Create TF-IDF matrix
        print("Vectorizing product descriptions...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.data[self.product_description].fillna(''))

        # Force garbage collection to free any temporary memory
        gc.collect()

        print(f"Indexed {self.tfidf_matrix.shape[0]} products with {self.tfidf_matrix.shape[1]} features")

        # self.tfidf_matrix.to_csv(self.output_file)

        return self

    def search(self, query, top_n=50):
        """
        Search for products matching the query without creating unnecessary copies.

        Parameters:
        -----------
        query : str
            User query
        top_n : int, default=5
            Number of top results to return

        Returns:
        --------
        pandas.DataFrame
            Top matching products with similarity scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")

        # Transform query to TF-IDF space
        query_vector = self.vectorizer.transform([query])

        # Calculate cosine similarity between query and all products
        # Using small batches if data is very large
        batch_size = 10000
        n_products = self.tfidf_matrix.shape[0]

        if n_products <= batch_size:
            # For smaller datasets, calculate all at once
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        else:
            # For larger datasets, calculate in batches to manage memory
            similarity_scores = np.zeros(n_products)
            for i in range(0, n_products, batch_size):
                print(f'Iteration: {i} for cosine similarity')
                end = min(i + batch_size, n_products)
                batch_scores = cosine_similarity(
                    query_vector,
                    self.tfidf_matrix[i:end]
                ).flatten()
                similarity_scores[i:end] = batch_scores

                # Force garbage collection after each batch
                gc.collect()

        # Get indices of top N products (using argpartition for efficiency)
        if top_n >= n_products:
            top_indices = similarity_scores.argsort()[::-1]
        else:
            # Use argpartition which is more efficient than argsort for partial sorting
            top_indices_unsorted = np.argpartition(similarity_scores, -top_n)[-top_n:]
            # Sort just the top indices
            top_indices = top_indices_unsorted[np.argsort(similarity_scores[top_indices_unsorted])][::-1]

        # Create minimal results DataFrame with only necessary data
        results = pd.DataFrame({
            self.product_key: self.product_keys[top_indices],
            'similarity_score': similarity_scores[top_indices]
        })

        # Add descriptions by reference to original data
        # This uses iloc which is more memory efficient than creating a new list
        results[self.product_description] = self.data.iloc[top_indices][self.product_description].values

        return results

    def get_feature_importance(self, query):
        """
        Get the most important terms for a given query.

        Parameters:
        -----------
        query : str
            User query

        Returns:
        --------
        dict
            Dictionary mapping terms to their importance scores
        """
        if self.tfidf_matrix is None:
            raise ValueError("Model has not been fit yet. Call fit() first.")

        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()

        # Transform query to get term weights
        query_vector = self.vectorizer.transform([query]).toarray()[0]

        # Create dictionary mapping terms to weights
        term_weights = {feature_names[i]: query_vector[i] for i in query_vector.nonzero()[0]}

        # Sort by weight in descending order
        sorted_terms = {k: v for k, v in sorted(term_weights.items(), key=lambda item: item[1], reverse=True)}

        return sorted_terms


# Example usage
if __name__ == "__main__":
    # Sample data
    sample_data = pd.DataFrame({
        'K_PRODUCT': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'D_PRODUCT': [
            'High performance laptop with 16GB RAM and 512GB SSD',
            'Gaming desktop computer with RTX 3080 graphics card',
            'Wireless bluetooth headphones with noise cancellation',
            'Ultra-wide 34-inch curved monitor for gaming and productivity',
            'Mechanical keyboard with RGB backlight and Cherry MX switches'
        ]
    })

    # Initialize and fit retriever
    retriever = TfidfRetriever(sample_data)

    # Search for products
    query = "gaming computer"
    results = retriever.search(query, top_n=3)
    print(f"Search results for '{query}':")
    print(results)

    # Get important terms
    important_terms = retriever.get_feature_importance(query)
    print("\nImportant terms in query:")
    for term, score in important_terms.items():
        print(f"{term}: {score:.4f}")