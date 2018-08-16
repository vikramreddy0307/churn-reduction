# churn-reduction
Churn (loss of customers to competition) is a problem for companies because it is moreexpensive to acquire a new customer than to keep your existing one from leaving.

# Introduction
# Problem statement:
This problem statement is targeted at enabling churn reduction using analytics concepts.
To develop an algorithm to predict the churn score based on usage
pattern.  

# Data
The predictors provided are as follows

* account length
* international plan
* voicemail plan
* number of voicemail messages
* total day minutes used
* day calls made
* total day charge
* total evening minutes
* total evening calls
* total evening charge
* total night minutes
* total night calls
* total night charge
* total international minutes used
* total international calls made
* total international charge
* number of customer service calls made

Target Variable :
move: if the customer has moved (1=yes; 0 = no)


# Exploratory data analysis:
* Missing values imputation
* Finding the Class distribution in target varible
* Correlation plot of the predictors in data
* Outlier analysis(imputation with median)

# Data preprocessing:
* Dealing with imbalanced data
* Removal for high correlated variables
* Converting to the required data types
* Removal of predictor variables
* Dealing with outliers


# Modelling:
* Model selection
* K-fold cross validation
* Applying different models
* Logistic regression
* Naive Bayes
* Random Forest
* K-Nearest neighbours


# Evaluation metrics:
* Predicting probabilities
* ROC curve

