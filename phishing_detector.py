import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

def prepare_dataset(data):
    # Handle missing values
    data.fillna(0, inplace=True)

    # Convert categorical variables to numeric (example: 'label' column)
    data['label'] = data['label'].map({'benign': 0, 'phishing': 1})  # Adjust according to your dataset's labels
    return data

def split_data(data):
    X = data.drop('label', axis=1)  # Features
    y = data['label']                # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

def main():
    dataset_path = "path/to/your/dataset.csv"  # Update with your actual dataset path
    dataset = load_dataset(dataset_path)
    prepared_data = prepare_dataset(dataset)
    X_train, X_test, y_train, y_test = split_data(prepared_data)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
