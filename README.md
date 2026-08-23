# BCG-Data-Science-Job-Simulation

Developed a machine-learning model to identify customers at high risk of churn, enabling a telecom company to target retention offers more effectively.

# Project Background

​"In this simulation, you’ll be working with a case team to help a client, PowerCo, investigate a problem with customer churn. PowerCo suspects price sensitivity is driving their customers to switch providers—the data tells a more complex story. Your job is to dig into the data, develop hypotheses, build predictive models, and translate your insights into strategic recommendations." (https://www.theforage.com/virtual-experience/Tcz8gTtprzAS4xSoK/bcg/data-science-ccdz/background-information?step=1)

## Insights and recommendations:

- Before offering discounts/deals to customers, we should identify those who have a higher chance of churning
- When making predictions, oversample the minority class (original dataset is heavily imbalanced)
- Rather than focusing on high accuracy, focus on the recall metric
- SMOTE + XGBClassifier seems to offer the best combination to achieve a high recall score for churners

## Key Results:
XGBoost + SMOTE achieved **75% recall** for the minority churn class, substantially improving identification of at-risk customers compared with the baseline Random Forest model.

# Data Structure 

We are working with a dataset consisting of 14,000+ records that has already been cleaned and has new features. The dataset contained significantly fewer churners than non-churners. To address this class imbalance, SMOTE was applied to the training data only, generating synthetic data of the minority class. This improved the model's ability to identify churners without artificially altering the validation/test distribution.

## Methodology

EDA - Train/test split - Baseline model (Random Forest) - XGBoost - SMOTE - GridSearchCV - Evaluation

# Executive Summary

## Overview of findings

- Random Forest Classifier with default parameters has a 0.06 recall rate for churners
- XGBClassifier has a 0.47 recall rate for churners
- XGBClassifier + SMOTE has a 0.75 recall rate for churners
- XGBClassifier + GridSearch optimized for F1 score has a 0.16 recall rate for churners
- XGBClassifier + GridSearch optimized for recall score has a 0.16 recall rate for churners


<img width="358" height="153" alt="image" src="https://github.com/user-attachments/assets/8cf6306b-b9d9-48f0-b5e5-12777e4161ba" />


<img width="386" height="155" alt="image" src="https://github.com/user-attachments/assets/6625872d-6bdc-46ad-a7ff-131fcc378a98" />


<img width="431" height="126" alt="image" src="https://github.com/user-attachments/assets/2c982a8b-164a-41d8-a784-cc28758cd545" />


<img width="475" height="230" alt="image" src="https://github.com/user-attachments/assets/2364e6ed-0eca-419e-af30-ad7c3254d9f9" />


<img width="476" height="227" alt="image" src="https://github.com/user-attachments/assets/9b7a350e-6927-4dad-bb67-b69ea003f6c0" />


# Insights Deep Dive

Because our goal is to correctly identify customers at risk of churning, the main metric to focus on is recall. While the Random Forest Classifier had a weighted average of 90% precision, the recall rate for churners was very poor, demonstrating how misleading precision can be when working with an imbalanced dataset. Using GridSearch to find the optimal models for F1 score and recall provided solid overall performance metrics, but both managed to only find 16% of all churners. XGBoost improved churn recall to 47% with decent overall metrics, while applying SMOTE increased recall to 75% and the worst overall metrics. This came with an increase in false positives, creating a tradeoff between identifying more potential churners and minimizing unnecessary discounts.

# Recommendations:
Based on the insights and findings above, we would recommend our client to consider the following:

Use the XGBClassifier + SMOTE model as the predictive model for customer retention. Its 75% recall means the model identifies significantly more potential churners than the alternatives. Although this results in more false positives and  some unnecessary discounts, the cost of these additional offers may be justified if retaining a customer is significantly more valuable than the cost of the discount.


## Technologies

Python, Pandas, Scikit-learn, XGBoost, GridSearchCV


