# Diabetes Prediction Project

## Introduction

This project aims to predict the likelihood of a person having diabetes using various machine learning models. The dataset contains medical and lifestyle-related features such as gender, age, hypertension, heart disease, smoking history, BMI, HbA1c level, and blood glucose level.

## Project Structure

The project is organized into several directories and files as follows:


### Directories and Files

- **data/**: Contains the dataset file (`diabetes_prediction_dataset.csv`).
- **models/**: Contains the trained models and the scaler object.
- **scripts/**: Contains the scripts for training, evaluating, and making predictions with the models.
- **notebooks/**: Contains Jupyter Notebooks for data analysis and model training.
- **README.md**: Provides an overview and instructions for the project.
- **requirements.txt**: Lists the dependencies required for the project.

## Dataset

The dataset contains the following columns:

- `gender`: Gender of the person (0: Female, 1: Male, 2: Other).
- `age`: Age of the person.
- `hypertension`: Whether the person has hypertension (0: No, 1: Yes).
- `heart_disease`: Whether the person has heart disease (0: No, 1: Yes).
- `smoking_history`: Smoking history of the person (0: Never, 1: Current, 2: Former/Ever/Not current, -1: Missing).
- `bmi`: Body Mass Index of the person.
- `HbA1c_level`: HbA1c level of the person.
- `blood_glucose_level`: Blood glucose level of the person.
- `diabetes`: Whether the person has diabetes (0: No, 1: Yes).

## Data Preprocessing

### Handling Missing Values

Missing values in the `smoking_history` column are replaced with `-1` to signify their absence. This approach ensures that all data points are accounted for, albeit with a designated placeholder value.

### Encoding Categorical Variables

The `gender` column is encoded as follows:
- Female: 0
- Male: 1
- Other: 2

The `smoking_history` column is encoded as follows:
- Never: 0
- Current: 1
- Former/Ever/Not current: 2
- Missing: -1

## Model Training

### Logistic Regression

Logistic Regression is used as one of the models to predict diabetes. It is a linear model suitable for binary classification problems.

### Random Forest

Random Forest is an ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

### Support Vector Machine (SVM)

SVM is a powerful classification method that finds the optimal hyperplane that best separates the classes in the feature space.

## Model Evaluation

The models are evaluated based on their accuracy, confusion matrix, and classification report. Below are the accuracies of the models:

- Logistic Regression: 0.96
- Random Forest: 0.97
- SVM: 0.96

## Prediction

A GUI application is created using Tkinter to allow users to input medical and lifestyle data and get a prediction on whether they have diabetes.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Train and evaluate models:
    ```bash
    python scripts/train_models.py
    python scripts/evaluate_models.py
    ```

4. Run the GUI for predictions:
    ```bash
    python scripts/gui_predict.py
    ```

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- tkinter
- joblib

## License

This project is licensed under the MIT License.
