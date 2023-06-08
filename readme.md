# Project Description

This [dataset](doc:https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) was retrieved from Kaggle.com. It contains the medical and demographic data from patients, along with their diabetes status.

# Project Goals

Using classification techniques, create a model that can accurately predict whether a patient has diabetes that performs better than baseline.


# Project Planning

Acquire
    - Retrieve data from Kaggle
    - 
Prepare
    - Split data into train, validate, test


Explore

Univariate Exploration
    - Histograms for numeric
    - Value Counts and Barplots for categorical

Bivariate Exploration
    - Split data into train, validate, test (stratify on diabetes)
    - Pairplot for numeric & numeric
    - Barplots for numeric & categorical

Multivariate Exploration
    - 

Hypothesize


Modeling
Pre-Processing
    - Encode object variables
    - Scale numeric variables

# Initial Hypothesis

My running hypothesis is that HbA1c levels will be the singular best predictor of whether someone has diabetes.

# Data Dictionary

Column Name | Description | Key
gender | Patient's Gender | Male, Female, Other
age | Patient's Age | Int
hypertension | Hypertension diagnosis | 0 = No, 1 = Yes
heart_disease | Heart disease diagnosis | 0 = No, 1 = Yes
smoking_history | Patient's smoking history | 
bmi | Patient's BMI score | Float
HbA1c_level | Patient's HbA1c level (average blood sugar level over the past 2-3 months) | Float
blood_glucose_level | Patient's blood glucose level (see note in notebook) | Float
diabetes | Diabetes diagnosis | 0 = No, 1 = Yes 

# How to Reproduce

Ensure that you have the csv file from Kaggle's website downloaded into your local repository. You will be able to run through the notebook without any additional credentials.

# Key Findings

TBD

# Recommendations

TBD

# Conclusion

TBD