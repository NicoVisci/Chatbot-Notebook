import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import plotly.express as px
import umap



class ProductClusterer:
    def __init__(self, product_df=None, input_file=None):
        """
        Initialize the clusterer with either a DataFrame or a file path.

        Args:
            product_df: Pandas DataFrame with product descriptions
            input_file: Path to CSV file with product data
        """
        if product_df is not None:
            self.product_df = product_df
        elif input_file is not None:
            self.product_df = pd.read_csv(input_file)
        else:
            raise ValueError("Either product_df or input_file must be provided")

        self.embeddings = None
        self.model = None
        self.cluster_labels = None
        self.reduced_embeddings = None

    def create_embeddings(self, model_name='all-MiniLM-L6-v2', output_file='data/DESCRIPTIONS_embedding.csv'):
        """
        Create embeddings for product descriptions using a pre-trained model.

        Args:
            model_name: Name of the SentenceTransformer model to use
            :param model_name:
            :param output_file:
        """
        if os.path.exists(output_file):
            # If it exists, load directly from the output file
            self.embeddings = pd.read_csv(output_file)
            print(f"Embeddings loaded from: {output_file}")
        else:
            print(f"Loading model: {model_name}")
            self.model = SentenceTransformer(model_name)

            descriptions = self.product_df['D_PRODUCT'].tolist()
            print(f"Creating embeddings for {len(descriptions)} descriptions")

            # Create embeddings
            self.embeddings = self.model.encode(descriptions, show_progress_bar=True)

            # Save embeddings to csv
            pd.DataFrame(self.embeddings.copy()).to_csv(output_file, index=False)

            return self.embeddings
        print(f"Embeddings shape: {self.embeddings.shape}")

    def cluster_kmeans(self, n_clusters=300):
        """
        Perform K-means clustering on the embeddings.

        Args:
            n_clusters: Number of clusters to create

        Returns:
            DataFrame with original data and cluster labels
        """
        if self.embeddings is not None:
            data = self.embeddings
        else:
            raise ValueError("Embeddings must be provided before clustering")


        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=3)
        self.cluster_labels = kmeans.fit_predict(data)
        print("KMeans clustering performed successfully")

        # Add cluster labels to the DataFrame
        result_df = self.product_df.copy()
        result_df['cluster'] = self.cluster_labels

        return result_df

    def cluster_dbscan(self, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering on the embeddings.

        Args:
            eps: The maximum distance between two samples for them to be considered as in the same neighborhood
            min_samples: The number of samples in a neighborhood for a point to be considered as a core point

        Returns:
            DataFrame with original data and cluster labels
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be created before clustering")

        # Perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = dbscan.fit_predict(self.embeddings)

        # Add cluster labels to the DataFrame
        result_df = self.product_df.copy()
        result_df['cluster'] = self.cluster_labels

        return result_df

    def find_optimal_clusters(self, max_clusters=300):
        """
        Find the optimal number of clusters using silhouette score.

        Args:
            max_clusters: Maximum number of clusters to try

        Returns:
            Optimal number of clusters
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be created before finding optimal clusters")

        silhouette_scores = []

        # Try different numbers of clusters
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(self.embeddings)

            # Calculate silhouette score
            score = silhouette_score(self.embeddings, cluster_labels)
            silhouette_scores.append(score)
            print(f"Clusters: {n_clusters}, Silhouette Score: {score:.4f}")

        # Find optimal number of clusters
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"Optimal number of clusters: {optimal_clusters}")

        return optimal_clusters

    def reduce_dimensions(self, method='pca', n_components=2):
        """
        Reduce the dimensionality of embeddings for visualization.

        Args:
            method: 'pca' or 'umap'
            n_components: Number of dimensions to reduce to

        Returns:
            Reduced embeddings
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be created before reducing dimensions")

        if method == 'pca':
            reducer = PCA(n_components=n_components)
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=n_components)
            self.reduced_embeddings = reducer.fit_transform(self.embeddings)

        return self.reduced_embeddings

    def visualize_clusters(self, reduction_method='pca'):
        """
        Visualize the clusters in 2D.

        Args:
            reduction_method: 'pca' or 'umap'
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before visualization")

        # Reduce dimensions for visualization if not already done
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 2:
            self.reduce_dimensions(method=reduction_method, n_components=2)


        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': self.reduced_embeddings[:, 0],
            'y': self.reduced_embeddings[:, 1],
            'cluster': self.cluster_labels,
            'description': self.product_df['D_PRODUCT'].values
        })

        # Plot with Plotly
        fig = px.scatter(
            plot_df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['description'],
            title=f'Product Clusters ({reduction_method.upper()} Reduction)'
        )
        fig.show()

    def get_cluster_summary(self):
        """
        Get a summary of each cluster.

        Returns:
            DataFrame with cluster statistics
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before getting summary")

        result_df = self.product_df.copy()
        result_df['cluster'] = self.cluster_labels

        # Get counts for each cluster
        cluster_counts = result_df['cluster'].value_counts().sort_index()

        # Sample products from each cluster
        cluster_samples = {}
        for cluster in cluster_counts.index:
            samples = result_df[result_df['cluster'] == cluster]['D_PRODUCT'].sample(
                min(3, len(result_df[result_df['cluster'] == cluster]))
            ).tolist()
            cluster_samples[cluster] = samples

        # Create summary DataFrame
        summary = pd.DataFrame({
            'cluster': cluster_counts.index,
            'count': cluster_counts.values,
            'percentage': cluster_counts.values / len(result_df) * 100,
            'samples': cluster_samples.values()
        })

        return summary

    def find_most_similar_products(self, query_text, top_n=5):
        """
        Find the most similar products to a query text.

        Args:
            query_text: Text to compare against product descriptions
            top_n: Number of similar products to return

        Returns:
            DataFrame with similar products
        """
        if self.model is None:
            raise ValueError("Model must be loaded before finding similar products")

        # Encode the query text
        query_embedding = self.model.encode([query_text])[0]

        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top similar products
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        # Create results DataFrame
        results = self.product_df.iloc[top_indices].copy()
        results['similarity'] = similarities[top_indices]

        return results
