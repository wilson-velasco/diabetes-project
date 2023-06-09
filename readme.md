![alt text](https://github.com/wilson-velasco/diabetes-project/blob/main/Exec_sum.png?raw=true)

# Project Description

This <a href='https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset'>dataset</a> was retrieved from Kaggle.com. It contains the medical and demographic data from patients, along with their diabetes status.

# Project Goals

Using classification techniques, create a model that can accurately predict whether a patient has diabetes that performs better than baseline.


# Project Planning

Acquire
    - Retrieve data from Kaggle, save file to local repository.
    - Check .head( ), .info( ), .describe( ), .shape.

Prepare
    - Check for nulls.
    - Fix age so it's rounded to the nearest whole number and converted to integer.
    - Drop smoking_history for MVP since No Info (i.e. null value) makes up 36% the dataset.
    - Given the oddly high amount of occurences, remove all BMI values that are exactly 27.32.
    - Reduce BMI to snip out top .05% of values.
    - Drop "Other" from gender since it's only .02% of dataset.
    - Keeping 0s and 1s in hypertension, heart disease, and diabetes for future modeling, will relabel to Yes and No for clarification during exploration on graphs alone.
    - Encode string values for gender for future modeling.
    - Split data into train, validate, test

Explore
    - Univariate Exploration (done during Prepare phase)
        - Histograms for numeric 
        - Value counts and barplots for categorical
    - Bivariate Exploration
        - Numeric variables against each other (Multicollinearity Check)
        - Numeric variables against target (Boxplots)
        - Categorical variables against target (Barplots)
    -Multivariate Exploration
        - Numeric variables against each other, with target variable set as Hue

Hypothesize
    - Run T-Test for numeric variables and categorical target.
    - Run Chi-squared for categorical variables and categorical target.

Modeling
    - Scale our data
    - Run through four classification models
        - Decision Tree
        - Random Forest
        - KNN
        - Logistic Regression
    - Run best model on test set

# Initial Hypothesis

My running hypothesis is that HbA1c levels will be the singular best predictor of whether someone has diabetes.

# Data Dictionary

Column Name | Description | Key
--- | --- | ---
gender | Patient's Gender | Male, Female, Other
age | Patient's Age | Int
hypertension | Hypertension diagnosis | 0 = No, 1 = Yes
heart_disease | Heart disease diagnosis | 0 = No, 1 = Yes
smoking_history | Patient's smoking history | (Removed)
bmi | Patient's BMI score | Float
HbA1c_level | Patient's HbA1c level (average blood sugar level over the past 2-3 months) | Float
blood_glucose_level | Patient's blood glucose level (see note in notebook) | Float
diabetes | Diabetes diagnosis | 0 = No, 1 = Yes 

# How to Reproduce

Ensure that you have the csv file from Kaggle's website downloaded into your local repository. You will be able to run through the notebook without any additional credentials.

# Key Findings

- All variables had varying degrees of statistically significant correlation to diabetes.
- A clear cluster is visible when looking at HbA1c, blood glucose levels, and diabetes.
- As all variables have some correlation, all will be sent into modeling. 
- In a nutshell, the older you are and the higher your HbA1c and blood glucose levels are, the more likely you are to have diabetes.
- Additionally, if you have hypertension and/or you have a heart disease, you are more likely to have diabetes.


![alt text](https://github.com/wilson-velasco/diabetes-project/blob/main/Conclusion.png?raw=true)


![alt text](https://github.com/wilson-velasco/diabetes-project/blob/main/Conclusion.png?raw=true)