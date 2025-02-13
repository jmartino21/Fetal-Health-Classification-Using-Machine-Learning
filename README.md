# Fetal Health Classification Using Machine Learning

## Overview
This project applies machine learning classification models to predict fetal health outcomes based on heart rate data. The models include Naive Bayes, Decision Tree, and Random Forest.

## Features
- **ASTV** – Acceleration Percentage
- **MLTV** – Mean Line Variability
- **Max** – Maximum heart rate
- **Median** – Median heart rate
- **NSP** – Class Label (1: Normal, 0: Abnormal)

## Models Implemented
- **Naive Bayes Classifier** (Gaussian Naive Bayes)
- **Decision Tree Classifier**
- **Random Forest Classifier** (Hyperparameter tuning included)

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Dataset
This project requires the **CTG - Raw Data.csv** dataset. Ensure it is placed in the same directory as the script. If missing, you can download it from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Cardiotocography).

## Usage
### Running the Script
Execute the script using:
```bash
python fetal_health_classification.py
```

### Steps Performed
1. Loads and preprocesses the dataset.
2. Splits the dataset into training and testing sets.
3. Trains Naive Bayes, Decision Tree, and Random Forest models.
4. Evaluates models using accuracy and confusion matrices.
5. Tunes Random Forest hyperparameters by testing various `n_estimators` values and graphing error rates.
6. Displays results and comparison.

## Output
- Accuracy scores for each classification model.
- Confusion matrices visualizing model performance.
- Hyperparameter tuning graph for Random Forest showing depth vs. number of estimators and corresponding error rates.

## License
This project is open-source and available for modification and use.

