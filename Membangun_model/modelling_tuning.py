import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import warnings
import sys


mlflow.set_registry_uri("http://127.0.0.1:500/")

mlflow.set_experiment("Submission Membangun Sistem Machine Learning - Alya FauziaSubmission")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(400)

    # Read file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dir = os.path.join(script_dir, "LoanEligibilityDataset_preprocessing")

    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(default_dir, "train_data.csv")
    data = pd.read_csv(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('Loan_Status', axis=1),
        data['Loan_Status'],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]

    # Definisi hyperparameter tuning dengan elastic search parameters
    n_neighbors_range = np.linspace(5, 50, 3, dtype=int)
    leaf_size_range = np.linspace(10, 100, 5, dtype=int)

    best_accuracy = 0
    best_params = []

    for n_neighbors in n_neighbors_range:
        for leaf_size in leaf_size_range:
            with mlflow.start_run(run_name=f"elastic_search_{n_neighbors}_{leaf_size}"):
                mlflow.autolog()

                # Train model
                model = KNeighborsClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size, metric='minkowski')
                model.fit(X_train, y_train)

                # Evaluate model
                accuracy = model.score(X_test, y_test)

                mlflow.log_metric("accuracy", accuracy)

                # Save best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {"n_neighbors": n_neighbors, "leaf_size":leaf_size}
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path = "model",
                        input_example=input_example
                    )