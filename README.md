# Travel Insurance
# Overview
The project inspects Travel Insurance Prediction Data from Kaggle. 

The primary objectives are to clean the data, perform exploratory data analysis, statistical analysis, and apply various machine learning models for target variable TravelInsurance prediction.

## Dataset
Dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/tejashvi14/travel-insurance-prediction-data).

## Python Libraries

This analysis was conducted using Python 3.11. The following packages were utilized:

- matplotlib=3.7.1
- numpy=1.25.0
- pandas=1.5.3
- scipy=1.11.1
- seaborn=0.12.2
- sklearn=0.0.post5
- statsmodels=0.14.0
- textblob=0.17.1
- unidecode=1.3.7

## Findings

•	Exploratory Data Analysis (EDA): The dataset made of 1987 observations and 9 features. Data is imbalanced and describes people in their 20s, not working for government, Graduated, no Chronic Diseases, not frequently flying, never been abroad.
•	Correlation: No 2 features are strong correlated, no feature pair with linear relationship.
•	Feature Engineering: Flying-related data (FrequentFlyer, EverTravelledAbroad) seem to be related to AnnualIncome. Those features were combined into a new feature. Weights were added by counting instance frequency.
•	Statistical Testing: Hypothesis testing revealed that AnnualIncome is to a degree related to Flying, Education, Health features.
•	Models: Various machine learning models (KNN, Support Vector Machines, Decision Tree, Random Forest, Naive Bayers) were tested, as well as Voting Classifiers. The best Model reaches an accuracy 79% (Radial Support Vector Machines)

## Suggestions for Insurance companies

* People who either fly frequently or have been abroad buy Travel Insurance in a much greater percentage. Travel insurance could be suggested for frequent flyers or together with abroad travel tickets.


## Future Work

- Explore the non-linear model Gradient Boosting to improve predictive performance.
- Employing dimensionality reduction techniques like Principal Component Analysis (PCA) or t-distributed Stochastic Neighbor Embedding (t-SNE) to condense the feature space and enhance interpretability.
- Address class imbalance by utilizing advanced methods, such as Synthetic Minority Over-sampling Technique (SMOTE), Adaptive Synthetic Sampling (ADASYN), or weighted loss functions within models.
