import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# step1: we have to load and explore the data
# Load the historical customer data

df = pd.read_csv ('C:/Users/jaffar/Downloads/Customer Churn_Modelling.csv')

# Perform data cleaning and feature engineering as needed
#df represents the data frame
# now the next step is to feature the selection
# Identify relevant features for churn prediction
# Examples may include usage behavior, customer demographics, contract length, etc.

# Feature Selection
features = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']]
target = df['Exited']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

#now the next step is to do the data scaling

#standardize the feature values to ensure the same scale

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# now the step is to model the training 
# Train various models - Logistic Regression, Random Forest, Gradient Boosting
# for better experience we can experiment with hyperparameter tuning for better performance

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # last step is to Evaluate the model:-
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\n{name} Model Accuracy: {accuracy:.2f}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')







