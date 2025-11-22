import mlflow
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import sys
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))

    # folder dataset
    preprocess_dir = os.path.join(base_dir, "LoanEligibilityDataset_preprocessing")

    # file yang ingin dibaca
    file_path = os.path.join(preprocess_dir, "train_data.csv")
    data = pd.read_csv(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop('Loan_Status', axis=1),
        data['Loan_Status'],
        random_state=42,
        test_size=0.2
    )

    input_example = X_train[0:5]
    n_neighbors = 5
    metric = 'minkowski'
    leaf_size = 30

    with mlflow.start_run():
        model = KNeighborsClassifier(n_neighbors=n_neighbors,
                                     metric=metric,
                                     leaf_size=leaf_size)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model = model,
            artifact_path = 'model',
            input_example=input_example
        )

        model.fit(X_train, y_train)

        # Metrics
        accuracy = model.score(X_test, y_test)

        # Log metrics ke MLflow
        mlflow.log_metric("accuracy", accuracy)
