# Project Description

This [dataset](doc:https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) was retrieved from Kaggle.com. It contains the medical and demographic data from patients, along with their diabetes status.

# Project Goals

Using classification techniques, create a model that can accurately predict whether a patient has diabetes that performs better than baseline.


# Project Planning

Acquire
    - Retrieve data from Kaggle, save file to local repository.
    - Check .head( ), .info( ), .describe( ), .shape.
Prepare
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
    - Run through four classification models
        - Decision Tree
        - Random Forest
        - KNN
        - Logistic Regression

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

# Conclusion

- Our best performing model, Decision Tree with max_depth of 10, outperformed our baseline by 7%, increasing accuracy to 97%.
- All features were found to have some degree of correlation, with HbA1c the strongest of the numeric variables, and hypertension the strongest of the categorical variables.

# Recommendations

- For the first iteration, I did not scale the data for the KNN neighbors. The second iteration should definitely address this, though I do not predict that it would perform better than our other models.
- BMI is a notoriously outdated measurement in the medical field, as it does not take into account a patient's body composition, i.e. how much lean muscle mass they have. Collecting data on body fat percentage would probably be a better predictor of diabetes.
- Blood glucose levels should be standardized only to include tests in which the patients are fasting. Glucose levels drastically increase after meals, which can throw off the data when it comes to predicting diabetes.
- A cluster was clear in the multivariate graph containing HbA1c level, blood glucose level, and diabetes. I would like to explore that in the second iteration.