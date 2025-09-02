# Decision Tree Classifier for UCI Letter Recognition Dataset

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.multiclass import OneVsRestClassifier

# Load dataset
file_path = 'letter.csv'
df = pd.read_csv(file_path)
print(df.info())

# Drop duplicate rows
df = df.drop_duplicates()

# Correlation heatmap
numerical_df = df.select_dtypes(include=np.number)
plt.figure(figsize=(12, 6))
sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Class distribution plot
plt.figure(figsize=(12, 6))
sns.countplot(x=df['class'])
plt.title("Class Distribution (A-Z)")
plt.xlabel("Letter Class")
plt.ylabel("Frequency")
plt.xticks(rotation=90)
plt.show()

# Split features and labels
X = df.drop(columns=['class'])
y = df['class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# Evaluation
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
plt.figure(figsize=(12, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = dt_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 6))
sns.barplot(x=X.columns[sorted_idx], y=importances[sorted_idx])
plt.xticks(rotation=90)
plt.title("Feature Importance")
plt.show()

# ROC curve (One-vs-Rest)
y_bin = label_binarize(y_test, classes=np.unique(y_train))
n_classes = y_bin.shape[1]
dt_model_ovr = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
dt_model_ovr.fit(X_train, y_train)
y_score = np.nan_to_num(dt_model_ovr.predict_proba(X_test))
y_score /= np.maximum(y_score.sum(axis=1, keepdims=True), 1e-9)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro-average ROC
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

plt.figure(figsize=(10, 6))
plt.plot(all_fpr, mean_tpr, color='blue', label='Macro-average ROC')
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], linestyle='--', label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Full tree visualization
plt.figure(figsize=(20, 10))
plot_tree(dt_model, feature_names=X.columns, class_names=np.unique(y), filled=True, fontsize=8)
plt.title("Full Decision Tree")
plt.show()

# Pruned tree training
pruned_model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
pruned_model.fit(X_train, y_train)
y_pred_pruned = pruned_model.predict(X_test)
print(f'Accuracy after Pruning: {accuracy_score(y_test, y_pred_pruned):.4f}')

# Pruned tree visualization
plt.figure(figsize=(20, 10))
plot_tree(pruned_model, feature_names=X.columns, class_names=np.unique(y), filled=True, fontsize=8)
plt.title("Pruned Decision Tree")
plt.savefig("pruned_decision_tree.png", dpi=300, bbox_inches='tight')
plt.show()
