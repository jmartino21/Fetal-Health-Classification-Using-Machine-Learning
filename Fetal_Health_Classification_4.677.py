import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset not found: {filename}. Please make sure the dataset is in the correct directory.")
    df = pd.read_csv(filename)
    df = df[["ASTV", "MLTV", "Max", "Median", "NSP"]]
    df = df.dropna()
    df['NSP'] = np.where(df['NSP'] == 1, 1, 0)
    return df

# Split dataset into training and test sets
def split_data(df, test_size=0.5, random_state=123):
    X = df[["ASTV", "MLTV", "Max", "Median"]].values
    Y = df[["NSP"]].values.ravel()
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Train and evaluate classifier
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Accuracy: {accuracy:.4f} ({accuracy:.0%})")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()
    return accuracy

# Main script execution
if __name__ == "__main__":
    file_path = 'CTG - Raw Data.csv'  # Ensure this file is present
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train and evaluate models
    models = {
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=7, max_depth=5, criterion='entropy', random_state=123)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test, name)
    
    print("Final Accuracy Scores:", results)
