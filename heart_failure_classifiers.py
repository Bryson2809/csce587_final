# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 18:03:28 2022

@authors: Bryson CarrollSSS
@email: jbc2@email.sc.edu

Purpose: We have created a decision tree along with a graph of the feature importances that
cause lead our program to decide whether or not a death event is going to take place for a patient.
We have also created a random forest classifier and we output a grapch of the ROC curve to show
the accuracy of our predictions.  
"""

import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, plot_roc_curve
import os

#import file and read into pandas dataframe
file_path = os.getcwd() + '\heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(file_path)
df = df.sample(frac=1)

#Create data for training and test sets, I decided the best aproach is splitting the data 80/20, whcih comes to 240 for train and 60 for test
train_df = df.iloc[0:240]
test_df = df.iloc[240:]

#Split into X and Y dataframes, X data are the features used to make a decision, and Y data is the feature being predicted (DEATH EVENT)
train_X = train_df[train_df.columns[:-1]]
train_Y = train_df['DEATH_EVENT']
test_X = test_df[test_df.columns[:-1]]
test_Y = test_df['DEATH_EVENT']

#Train and plot decision tree classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_X, train_Y)
plt.figure(figsize=(30, 15))
tree.plot_tree(clf, fontsize=12)

#Binary predictions for death states, prints to matrix
test_predict_Y = clf.predict(test_X)

#Print results of the predictions as well as the confusion matrix
print("Predictions: ", test_predict_Y)
print("Confusion matrix: ", confusion_matrix(test_Y, test_predict_Y, labels=[1,0]))

#Calculate tp, fn, fp, and fn
#Print the classification report from sklearn
tp, fn, fp, tn = confusion_matrix(test_Y,test_predict_Y,labels=[1,0]).reshape(-1)
print('True Positives: ',tp)
print('False Negatives: ',fn)
print('False Positives: ',fp)
print('True Negatives: ',tn)

print('Classification Report: \n', classification_report(test_Y,test_predict_Y,labels=[1,0]))

# print feature importances plot
plt.figure()
plt.style.use("ggplot")
data = clf.feature_importances_
x_pos = [i for i, _ in enumerate(data)]
plt.bar(x_pos, data, color="blue")
plt.xlabel("Feature Number")
plt.title("Feature importances")
plt.xticks(x_pos)
plt.show()

#Radnom forest classifier
print('\nRandom Forest__________________________________')

#Train random forest classifier and print the resultant predictions
rf_classifier = RandomForestClassifier(n_estimators=150)
pipe = make_pipeline(rf_classifier)
pipe.fit(train_X, train_Y)
pred_Y = pipe.predict(test_X)
print(f"Predicted values: ", pred_Y)
confusion = confusion_matrix(test_Y, test_predict_Y, labels=[1,0])

#Print confusion matrix
print(confusion)

#Print accuracy results
tp, fn, fp, tn = confusion.reshape(-1)
print('True Positives: ',tp)
print('False Negatives: ', fn)
print('False Positives: ', fp)
print('True Negatives: ', tn)

print('Classification Report: \n', classification_report(test_Y, test_predict_Y,labels=[1,0]))

# ROC AUC score calculation and print ROC curve
train_probs = pipe.predict_proba(train_X)[:, 1]
probs = pipe.predict_proba(test_X)[:, 1]
train_predictions = pipe.predict(train_X)
print(f"Train ROC AUC Score: {roc_auc_score(train_Y, train_probs)}")
print(f"Test ROC AUC Score: {roc_auc_score(test_Y, probs)}")
plot_roc_curve(pipe, test_X, test_Y)
plt.xlabel('False Positive Rate (Has DEATH_EVENT: 1)')
plt.ylabel('True Positive Rate (Has DEATH_EVENT: 1)')
plt.show()
