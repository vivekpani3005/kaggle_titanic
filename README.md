# Overview
This repository contains code for predicting passenger survival on the Titanic using machine learning techniques. The goal is to analyze the provided dataset, build a predictive model using logistic regression, and submit predictions for the test data to Kaggle.

# Dataset
The dataset consists of two CSV files:

train.csv: The training data with features and corresponding survival labels.
test.csv: The test data with features for which predictions need to be made.
# Code Files
titanic_survival_prediction.ipynb: This Jupyter Notebook contains all the code for data cleaning, preprocessing, model training, and prediction generation.
# Requirements
The code is written in Python, and you will need the following libraries:

* pandas
* scikit-learn
You can install the required libraries using pip:

>pip install pandas scikit-learn

# Running the Code
1. Clone the repository to your local machine.
1. Place the train.csv and test.csv files in the same directory as the Jupyter Notebook.
1. Open titanic_survival_prediction.ipynb in Jupyter Notebook.
1. Run the cells sequentially to execute the code step-by-step.
# Data Cleaning and Preprocessing
The clean function is used to clean the data. It removes unnecessary columns like Ticket, Cabin, Name, and PassengerId. Missing values in numerical columns (SibSp, Parch, Fare, and Age) are filled with their respective median values. The Embarked column is filled with "U" for unknown values. Categorical columns (Sex and Embarked) are label-encoded using LabelEncoder from scikit-learn.

# Model Training
A logistic regression model is used for predicting survival. The training data is split into training and validation sets using the train_test_split function from scikit-learn. The logistic regression model is then trained on the training set.

# Model Evaluation
The model's accuracy is evaluated on the validation set using the accuracy_score function from scikit-learn.

# Generating Predictions
The trained logistic regression model is used to make predictions on the test data. The predictions are then saved in a CSV file named submission.csv, which is formatted with the required columns (PassengerId and Survived) for submission to Kaggle.

# Kaggle Submission
The generated submission.csv file can be uploaded to Kaggle to evaluate the model's performance on the test data.

# Disclaimer
This project is for educational purposes only and is not meant for production use. The provided dataset is from Kaggle's Titanic competition, and the goal is to practice data analysis and machine learning techniques.

Feel free to use and modify the code according to your needs. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request.

Happy coding!
