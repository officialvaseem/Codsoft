
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/jaffar/Downloads/credit_card_fraud_detection.csv')



print(df.head())
print(df.info())
print(df.isnull().sum())

X = df.drop('Class', axis=1)  # Assuming 'Class' is the column indicating fraud or not
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Logistic Regression
print("Logistic Regression:")
log_reg_model = LogisticRegression()
train_evaluate_model(log_reg_model, X_train, y_train, X_test, y_test)

# Decision Tree
print("\nDecision Tree:")
dt_model = DecisionTreeClassifier(random_state=42)
train_evaluate_model(dt_model, X_train, y_train, X_test, y_test)

# Random Forest
print("\nRandom Forest:")
rf_model = RandomForestClassifier(random_state=42)
train_evaluate_model(rf_model, X_train, y_train, X_test, y_test)
