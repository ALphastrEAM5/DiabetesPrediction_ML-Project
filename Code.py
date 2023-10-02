import numpy as np
import warnings
import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures
import shap  # Import SHAP
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('diabetes.csv')


#Calculate key statistics for the dataset
dataset_summary = data.describe()

# Display key statistics
print("Dataset Summary:")
print(dataset_summary)



# Create polynomial features for selected features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(data[['Glucose', 'BMI', 'Age']])

# Manually generate feature names for polynomial features
poly_feature_names = [f'{col1}_x_{col2}' for col1 in ['Glucose', 'BMI', 'Age'] for col2 in ['Glucose', 'BMI', 'Age']]
poly_data = pd.DataFrame(poly_features, columns=poly_feature_names)

data2 = pd.concat([data, poly_data], axis=1)


# Create interaction terms for selected pairs of features
data2['Glucose_BMI'] = data['Glucose'] * data['BMI']
data2['Age_Pregnancies'] = data['Age'] * data['Pregnancies']


# Create a new DataFrame with the interaction terms and Outcome
interaction_df = pd.DataFrame({
    'Glucose_BMI': data2['Glucose_BMI'],
    'Age_Pregnancies': data2['Age_Pregnancies'],
    'Outcome': data['Outcome']
})

# Create a contingency table for Glucose_BMI vs. Outcome
glucose_bmi_table = pd.crosstab(interaction_df['Glucose_BMI'], interaction_df['Outcome'])

# Create a contingency table for Age_Pregnancies vs. Outcome
age_pregnancies_table = pd.crosstab(interaction_df['Age_Pregnancies'], interaction_df['Outcome'])

# Format the tables using tabulate
formatted_glucose_bmi_table = tabulate(glucose_bmi_table, headers='keys', tablefmt='fancy_grid')
formatted_age_pregnancies_table = tabulate(age_pregnancies_table, headers='keys', tablefmt='fancy_grid')

# Display the formatted tables
print("2x2 Contingency Table for Glucose_BMI vs. Outcome:")
print(formatted_glucose_bmi_table)

print("\n2x2 Contingency Table for Age_Pregnancies vs. Outcome:")
print(formatted_age_pregnancies_table)

# Plot Gaussian distribution bell curves
plt.figure(figsize=(12, 6))

# Plot Glucose_BMI bell curve
plt.subplot(1, 2, 1)
sns.histplot(interaction_df['Glucose_BMI'], kde=True, color='blue', bins=30, label='Glucose_BMI')
plt.title('Gaussian Distribution of Glucose_BMI')
plt.xlabel('Glucose_BMI')
plt.ylabel('Frequency')
plt.legend()

# Plot Age_Pregnancies bell curve
plt.subplot(1, 2, 2)
sns.histplot(interaction_df['Age_Pregnancies'], kde=True, color='green', bins=30, label='Age_Pregnancies')
plt.title('Gaussian Distribution of Age_Pregnancies')
plt.xlabel('Age_Pregnancies')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# Prepare the data
X_original = data.drop(columns='Outcome', axis=1)
Y_original = data['Outcome']

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X_original)

X = X_standardized
Y = Y_original

# Visualize the distribution of the target variable 'Outcome'
plt.figure(figsize=(8, 4))
sns.countplot(x='Outcome', data=data)
plt.title('Distribution of Diabetes Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Hyperparameter tuning using RandomizedSearchCV for SVM
param_dist_svm = {
    'C': np.logspace(-3, 3, 100),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': np.logspace(-3, 3, 100),
}

random_search_svm = RandomizedSearchCV(SVC(), param_distributions=param_dist_svm, n_iter=10, cv=5, n_jobs=1)
random_search_svm.fit(X_train, Y_train)

best_svm_classifier = random_search_svm.best_estimator_

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Train a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=2)
decision_tree_classifier.fit(X_train, Y_train)

# Train a Random Forest classifier
random_forest_classifier = RandomForestClassifier(random_state=2)
random_forest_classifier.fit(X_train, Y_train)

# Evaluate SVM model using k-fold cross-validation
svm_scores = cross_val_score(best_svm_classifier, X_train, Y_train, cv=5)
print('SVM - Cross-Validation Scores:', svm_scores)
print('SVM - Mean Cross-Validation Score:', np.mean(svm_scores))


# Create a new dataset without individual Glucose and BMI columns
data3 = data.drop(['Glucose', 'BMI'], axis=1)

# Add the 'Glucose_BMI' column to data3
data3['Glucose_BMI'] = data2['Glucose_BMI']

# Prepare the data
X_original_3 = data3.drop(columns='Outcome', axis=1)
Y_original_3 = data3['Outcome']

scaler_3 = StandardScaler()
X_standardized_3 = scaler_3.fit_transform(X_original_3)

X_3 = X_standardized_3
Y_3 = Y_original_3

# Split the new data (data3) into training and testing sets
X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(X_3, Y_3, test_size=0.2, stratify=Y_3, random_state=2)

# Hyperparameter tuning using RandomizedSearchCV for the new SVM classifier
random_search_svm_3 = RandomizedSearchCV(SVC(), param_distributions=param_dist_svm, n_iter=10, cv=5, n_jobs=1)
random_search_svm_3.fit(X_train_3, Y_train_3)

best_svm_classifier_3 = random_search_svm_3.best_estimator_

# Evaluate the new SVM model using k-fold cross-validation
svm_scores_3 = cross_val_score(best_svm_classifier_3, X_train_3, Y_train_3, cv=5)
print('New SVM - Cross-Validation Scores:', svm_scores_3)
print('New SVM - Mean Cross-Validation Score:', np.mean(svm_scores_3))

# Predictions and accuracy for the new SVM classifier
Y_train_pred_svm_3 = best_svm_classifier_3.predict(X_train_3)
training_accuracy_svm_3 = accuracy_score(Y_train_pred_svm_3, Y_train_3)
recall_svm_3 = recall_score(Y_train_3, Y_train_pred_svm_3)
f1_svm_3 = f1_score(Y_train_3, Y_train_pred_svm_3)
roc_auc_svm_3 = roc_auc_score(Y_train_3, best_svm_classifier_3.decision_function(X_train_3))
print('New SVM - Accuracy Score on Training Data: ', training_accuracy_svm_3)
print('New SVM - Recall on Training Data: ', recall_svm_3)
print('New SVM - F1 Score on Training Data: ', f1_svm_3)
print('New SVM - ROC AUC on Training Data: ', roc_auc_svm_3)

Y_test_pred_svm_3 = best_svm_classifier_3.predict(X_test_3)
test_accuracy_svm_3 = accuracy_score(Y_test_pred_svm_3, Y_test_3)
recall_svm_test_3 = recall_score(Y_test_3, Y_test_pred_svm_3)
f1_svm_test_3 = f1_score(Y_test_3, Y_test_pred_svm_3)
roc_auc_svm_test_3 = roc_auc_score(Y_test_3, best_svm_classifier_3.decision_function(X_test_3))

print('New SVM - Accuracy Score on Test Data: ', test_accuracy_svm_3)
print('New SVM - Recall on Test Data: ', recall_svm_test_3)
print('New SVM - F1 Score on Test Data: ', f1_svm_test_3)
print('New SVM - ROC AUC on Test Data: ', roc_auc_svm_test_3)





# Create a new dataset without individual Glucose, Age and BMI columns
data4 = data.drop(['Glucose', 'BMI', 'Age'], axis=1)

# Add the 'Polynomial_term' column to data4
data4['Polynomial_term'] = data2['Glucose_x_Glucose'] + data2['BMI_x_BMI'] + data2['Glucose_x_Age']

# Prepare the data
X_original_4 = data4.drop(columns='Outcome', axis=1)
Y_original_4 = data4['Outcome']

scaler_4 = StandardScaler()
X_standardized_4 = scaler_4.fit_transform(X_original_4)

X_4 = X_standardized_4
Y_4 = Y_original_4

# Split the new data (data3) into training and testing sets
X_train_4, X_test_4, Y_train_4, Y_test_4 = train_test_split(X_4, Y_4, test_size=0.2, stratify=Y_4, random_state=2)

# Hyperparameter tuning using RandomizedSearchCV for the new SVM classifier
random_search_svm_4 = RandomizedSearchCV(SVC(), param_distributions=param_dist_svm, n_iter=10, cv=5, n_jobs=1)
random_search_svm_4.fit(X_train_4, Y_train_4)

best_svm_classifier_4 = random_search_svm_4.best_estimator_

# Evaluate the new SVM model using k-fold cross-validation
svm_scores_4 = cross_val_score(best_svm_classifier_4, X_train_4, Y_train_4, cv=5)
print('Poly SVM - Cross-Validation Scores:', svm_scores_4)
print('Poly SVM - Mean Cross-Validation Score:', np.mean(svm_scores_4))

# Predictions and accuracy for the new SVM classifier
Y_train_pred_svm_4 = best_svm_classifier_4.predict(X_train_4)
training_accuracy_svm_4 = accuracy_score(Y_train_pred_svm_4, Y_train_4)
recall_svm_4 = recall_score(Y_train_4, Y_train_pred_svm_4)
f1_svm_4 = f1_score(Y_train_4, Y_train_pred_svm_4)
roc_auc_svm_4 = roc_auc_score(Y_train_4, best_svm_classifier_4.decision_function(X_train_4))
print('Poly SVM - Accuracy Score on Training Data: ', training_accuracy_svm_4)
print('Poly SVM - Recall on Training Data: ', recall_svm_4)
print('Poly SVM - F1 Score on Training Data: ', f1_svm_4)
print('Poly SVM - ROC AUC on Training Data: ', roc_auc_svm_4)

Y_test_pred_svm_4 = best_svm_classifier_4.predict(X_test_4)
test_accuracy_svm_4 = accuracy_score(Y_test_pred_svm_4, Y_test_4)
recall_svm_test_4 = recall_score(Y_test_4, Y_test_pred_svm_4)
f1_svm_test_4 = f1_score(Y_test_4, Y_test_pred_svm_4)
roc_auc_svm_test_4 = roc_auc_score(Y_test_4, best_svm_classifier_4.decision_function(X_test_4))

print('Poly SVM - Accuracy Score on Test Data: ', test_accuracy_svm_4)
print('Poly SVM - Recall on Test Data: ', recall_svm_test_4)
print('Poly SVM - F1 Score on Test Data: ', f1_svm_test_4)
print('Poly SVM - ROC AUC on Test Data: ', roc_auc_svm_test_4)





# Predictions and accuracy for SVM
Y_train_pred_svm = best_svm_classifier.predict(X_train)
training_accuracy_svm = accuracy_score(Y_train_pred_svm, Y_train)
recall_svm = recall_score(Y_train, Y_train_pred_svm)
f1_svm = f1_score(Y_train, Y_train_pred_svm)
roc_auc_svm = roc_auc_score(Y_train, best_svm_classifier.decision_function(X_train))
print('Best SVM - Accuracy Score on Training Data: ', training_accuracy_svm)
print('Best SVM - Recall on Training Data: ', recall_svm)
print('Best SVM - F1 Score on Training Data: ', f1_svm)
print('Best SVM - ROC AUC on Training Data: ', roc_auc_svm)

Y_test_pred_svm = best_svm_classifier.predict(X_test)
test_accuracy_svm = accuracy_score(Y_test_pred_svm, Y_test)
recall_svm_test = recall_score(Y_test, Y_test_pred_svm)
f1_svm_test = f1_score(Y_test, Y_test_pred_svm)
roc_auc_svm_test = roc_auc_score(Y_test, best_svm_classifier.decision_function(X_test))

print('Best SVM - Accuracy Score on Test Data: ', test_accuracy_svm)
print('Best SVM - Recall on Test Data: ', recall_svm_test)
print('Best SVM - F1 Score on Test Data: ', f1_svm_test)
print('Best SVM - ROC AUC on Test Data: ', roc_auc_svm_test)

# Predictions and accuracy for Decision Tree
Y_train_pred_dt = decision_tree_classifier.predict(X_train)
training_accuracy_dt = accuracy_score(Y_train_pred_dt, Y_train)
recall_dt = recall_score(Y_train, Y_train_pred_dt)
f1_dt = f1_score(Y_train, Y_train_pred_dt)
roc_auc_dt = roc_auc_score(Y_train, decision_tree_classifier.predict_proba(X_train)[:, 1])
print('Decision Tree - Accuracy Score on Training Data: ', training_accuracy_dt)
print('Decision Tree - Recall on Training Data: ', recall_dt)
print('Decision Tree - F1 Score on Training Data: ', f1_dt)
print('Decision Tree - ROC AUC on Training Data: ', roc_auc_dt)

Y_test_pred_dt = decision_tree_classifier.predict(X_test)
test_accuracy_dt = accuracy_score(Y_test_pred_dt, Y_test)
recall_dt_test = recall_score(Y_test, Y_test_pred_dt)
f1_dt_test = f1_score(Y_test, Y_test_pred_dt)
roc_auc_dt_test = roc_auc_score(Y_test, decision_tree_classifier.predict_proba(X_test)[:, 1])

print('Decision Tree - Accuracy Score on Test Data: ', test_accuracy_dt)
print('Decision Tree - Recall on Test Data: ', recall_dt_test)
print('Decision Tree - F1 Score on Test Data: ', f1_dt_test)
print('Decision Tree - ROC AUC on Test Data: ', roc_auc_dt_test)

# Predictions and accuracy for Random Forest
Y_train_pred_rf = random_forest_classifier.predict(X_train)
training_accuracy_rf = accuracy_score(Y_train_pred_rf, Y_train)
recall_rf = recall_score(Y_train, Y_train_pred_rf)
f1_rf = f1_score(Y_train, Y_train_pred_rf)
roc_auc_rf = roc_auc_score(Y_train, random_forest_classifier.predict_proba(X_train)[:, 1])
print('Random Forest - Accuracy Score on Training Data: ', training_accuracy_rf)
print('Random Forest - Recall on Training Data: ', recall_rf)
print('Random Forest - F1 Score on Training Data: ', f1_rf)
print('Random Forest - ROC AUC on Training Data: ', roc_auc_rf)

Y_test_pred_rf = random_forest_classifier.predict(X_test)
test_accuracy_rf = accuracy_score(Y_test_pred_rf, Y_test)
recall_rf_test = recall_score(Y_test, Y_test_pred_rf)
f1_rf_test = f1_score(Y_test, Y_test_pred_rf)
roc_auc_rf_test = roc_auc_score(Y_test, random_forest_classifier.predict_proba(X_test)[:, 1])
print('Random Forest - Accuracy Score on Test Data: ', test_accuracy_rf)
print('Random Forest - Recall on Test Data: ', recall_rf_test)
print('Random Forest - F1 Score on Test Data: ', f1_rf_test)
print('Random Forest - ROC AUC on Test Data: ', roc_auc_rf_test)

# Randomly sample 100 data points from X_test
sample_indices = random.sample(range(X_test.shape[0]), 100)
X_test_sampled = X_test[sample_indices]

# Feature Importance using SHAP for Random Forest
explainer = shap.TreeExplainer(random_forest_classifier)
shap_values = explainer.shap_values(X_test_sampled)



# Visualize feature importance using a summary plot
shap.summary_plot(shap_values, X_test_sampled, feature_names=data.columns)

# Get feature importances from the trained Random Forest classifier
feature_importances = random_forest_classifier.feature_importances_

# Create a DataFrame to associate feature names with their importances
feature_importance_df = pd.DataFrame({'Feature': data.columns[:-1], 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the sorted feature importances
print("Feature Importances:")
print(feature_importance_df)

selected_poly_features = ['Glucose_x_Glucose', 'BMI_x_BMI', 'Age_x_Age', 'BMI_x_Glucose', 'Glucose_x_Age', 'BMI_x_Age']

#plt.figure(figsize=(10, 6))

# Create subplots with a single shared Y-axis
fig, ax = plt.subplots(figsize=(10, 6))

for feature in selected_poly_features:
    sns.regplot(x=feature, y='Outcome', data=data2, logistic=True, ci=None, scatter_kws={'alpha': 0.3}, ax=ax, label=feature)

plt.title('Relationships for Polynomial & Interaction Terms vs. Diabetes Outcome')
plt.xlabel('Polynomial & Interaction Term')
plt.ylabel('Diabetes Outcome (0: Non-Diabetic, 1: Diabetic)')

plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Make predictions for new input data

a= float(input("Pregnancies? "))
b= float(input("GLucose? "))
c= float(input("Blood Pressure? "))
d= float(input("Skin Thickness? "))
e= float(input("Insulin? "))
f= float(input("BMI? "))
g= float(input("Diabetes Pedigree Function? "))
h= float(input("Age? "))
new_input = np.array([a, b, c, d, e, f, g, h]).reshape(1, -1)
new_input_standardized = scaler.transform(new_input)

# Predict using the Decision Tree classifier
dt_prediction = decision_tree_classifier.predict(new_input_standardized)

# Predict using the Random Forest classifier
rf_prediction = random_forest_classifier.predict(new_input_standardized)

# Predict using the Best SVM classifier
best_svm_prediction = best_svm_classifier.predict(new_input_standardized)

print("Best SVM Prediction:", best_svm_prediction[0])

print("Decision Tree Prediction:", dt_prediction[0])
print("Random Forest Prediction:", rf_prediction[0])




# Visualize the distribution of the target variable 'Outcome' using a pie chart
plt.figure(figsize=(8, 4))
plt.pie(data['Outcome'].value_counts(), labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Diabetes Outcome')
plt.show()

# Create a bar plot for cross-validation scores of SVM
plt.figure(figsize=(8, 4))
plt.bar(range(len(svm_scores)), svm_scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores for SVM')
plt.show()

# Create a radar chart for evaluation metrics of different models
metrics = ['Accuracy', 'Recall', 'F1 Score', 'ROC AUC']
svm_metrics = [test_accuracy_svm, recall_svm_test, f1_svm_test, roc_auc_svm_test]
dt_metrics = [test_accuracy_dt, recall_dt_test, f1_dt_test, roc_auc_dt_test]
rf_metrics = [test_accuracy_rf, recall_rf_test, f1_rf_test, roc_auc_rf_test]

# Ensure all metrics lists have the same length
assert len(metrics) == len(svm_metrics) == len(dt_metrics) == len(rf_metrics), "All lists must have the same length."

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()

fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

# Add a plot for each model's metrics
ax.fill(angles, svm_metrics, 'b', alpha=0.1, label='SVM')
ax.fill(angles, dt_metrics, 'r', alpha=0.1, label='Decision Tree')
ax.fill(angles, rf_metrics, 'g', alpha=0.1, label='Random Forest')

# Set the number of ticks and tick labels to match the number of metrics
ax.set_thetagrids(np.degrees(angles), metrics)

plt.title('Model Comparison Radar Chart')
plt.legend(loc='upper right')
plt.show()