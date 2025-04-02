import os.path
from setfit import SetFitModel, TrainingArguments, Trainer
from datasets import Dataset

import pandas as pd

from source_code.ConfigManager import ConfigManager


class IntentModule:
    """
    A module for training and using SetFit-based intent classification models.

    This class handles loading, training, and inference of intent recognition models
    using the SetFit framework for few-shot text classification. It manages dataset
    loading, model training/loading, and prediction functionality.
    """

    def __init__(self, dataset_name: str, labels, model_name="BAAI/bge-small-en-v1.5"):
        """
        Initialize an IntentModule with dataset information and model configuration.

        Args:
            dataset_name (str): Name of the dataset to use for training/evaluation
            labels (list): List of intent labels the model should predict
            model_name (str, optional): The base model to use from Hugging Face.
                                               Defaults to "BAAI/bge-small-en-v1.5".
        """
        self.trainer = None
        self.model = None
        self.trained = True

        self.modelPathName = model_name
        self.datasetName = dataset_name
        self.labels = labels

        self.intent_recognition_load()

    def intent_recognition_load(self):
        """
        Load datasets and initialize the intent recognition model.

        This method:
        1. Loads training, evaluation, and test datasets from CSV files
        2. Attempts to load a pre-trained model from local storage
        3. Falls back to loading a base model from Hugging Face if local model doesn't exist
        4. Sets up the training configuration and initializes the SetFit trainer
        5. Trains the model if a pre-trained version wasn't loaded
        6. Evaluates the model if debug mode is enabled in ConfigManager
        """
        # Load the dataset from the CSV file
        folder = 'data/intent_recognition_datasets/'
        train_dataset = pd.read_csv(folder + self.datasetName + '_Dataset.csv')
        train_dataset = Dataset.from_pandas(train_dataset)
        eval_dataset = pd.read_csv(folder + self.datasetName + '_Dataset_eval.csv')
        eval_dataset = Dataset.from_pandas(eval_dataset)
        test_dataset = pd.read_csv(folder + self.datasetName + '_Dataset_test.csv')
        test_dataset = Dataset.from_pandas(test_dataset)

        path = 'models/' + self.modelPathName + "_" + self.datasetName + '_pretrained'
        if os.path.exists(path):
            # Load a SetFit model from local
            self.model = SetFitModel.from_pretrained(
                path,
                labels=self.labels
            )
        if self.model is None:
            # Load a SetFit model from Hub
            self.model = SetFitModel.from_pretrained(
                self.modelPathName,
                labels=self.labels,
            )
            self.trained = False

        args = TrainingArguments(
            batch_size=16,
            num_epochs=4,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metric="accuracy",
            column_mapping={"Sentence": "text", "Intent": "label"}
            # Map dataset columns to text/label expected by trainer
        )

        # Train and evaluate
        if not self.trained:
            self.trainer.train()
            self.model.save_pretrained(path)
        if ConfigManager.CONFIG.get('debug', False):
            metrics = self.trainer.evaluate(test_dataset)
            print(metrics)

    def predict(self, text):
        """
        Predict the intent of the given text using the trained model.

        Args:
            text (str): The input text to classify

        Returns:
            str: The predicted intent label from the labels list provided during initialization
        """
        return self.model.predict(text)