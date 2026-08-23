# BCG-Data-Science-Job-Simulation

# Project Background
You will work to help a client analyze their problem with customer churn. Use data analysis and predictive modeling to identify customers who are at high-risk of churning.

Insights and recommendations:

- Before offering discounts/deals to customers, we should identify those who have a higher chance of churning
- When making predictions, oversample the minority class (original dataset is heavily imbalanced)
- SMOTE + XGBClassifier seems to offer the best combination to achieve a high recall score for churners

# Data Structure & Initial Checks

We are working with a dataset consisting of 14,000+ records that has already been cleaned and has new features.

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

Because our goal is to correctly identify customers at risk of churning, the main metric to focus on is recall. While the Random Forest Classifier had a weighted average of 90% accuracy, the recall rate for churners was very poor. Using GridSearch to find the optimal models for F! score and recall provided good overall performance metrics, but both managed to only find 16% of all churners. Using XGBClassifier provided overall decent metrics, and increased recall to nearly 50%. XGBClassifier + GridSearch has the worst overall metrics, however it correctly identifies 75% of all churners, which is the goal of this project. 

# Recommendations:
Based on the insights and findings above, we would recommend our client to consider the following:

Utilize the 75% recall rate model over the others. In this case, precision should not be the main focus. Although the model identifies many false positives, the tradeoff to identify as many churners as possible may be worth it. An argument can be made that offering more "unnecessary" discounts can lead to long-term customer retention, rather than correctly identifying churners to minimize the number of discounts given out.



