"""
Support Vector Machine (SVM) Classification Example
---------------------------------------------------
This script demonstrates the full Machine Learning lifecycle using
an SVM classifier on a sample dataset (Iris dataset).
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_data():
    """Load the Iris dataset as a sample classification problem."""
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names
    print(f"Dataset loaded with {X.shape[0]} samples and {X.shape[1]} features.")
    return X, y, target_names


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Split into train/test and scale features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data preprocessing complete. Features scaled.")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train, kernel="rbf", C=1.0, gamma="scale"):
    """Train an SVM classifier."""
    svm_clf = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    svm_clf.fit(X_train, y_train)
    print("Model training complete.")
    return svm_clf


def evaluate_model(model, X_test, y_test, target_names):
    """Evaluate model performance on the test set."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


def save_model(model, scaler, filename="svm_model.pkl"):
    """Save the trained model and scaler for future use."""
    joblib.dump({"model": model, "scaler": scaler}, filename)
    print(f"Model and scaler saved to {filename}.")


def load_model(filename="svm_model.pkl"):
    """Load the trained model and scaler."""
    data = joblib.load(filename)
    print(f"Model loaded from {filename}.")
    return data["model"], data["scaler"]


def main():
    # Step 1: Load data
    X, y, target_names = load_data()

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

    # Step 3: Train model
    model = train_model(X_train, y_train)

    # Step 4: Evaluate model
    evaluate_model(model, X_test, y_test, target_names)

    # Step 5: Save model
    save_model(model, scaler)

    # Step 6: Load model and re-test
    loaded_model, loaded_scaler = load_model()
    X_test_scaled = loaded_scaler.transform(X_test)
    print("\nRe-evaluating loaded model:")
    evaluate_model(loaded_model, X_test_scaled, y_test, target_names)


if __name__ == "__main__":
    main()
