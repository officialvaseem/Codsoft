import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    df = pd.read_csv('C:/Users/jaffar/Downloads/spamdetection.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('C:/Users/jaffar/Downloads/spamdetection.csv', encoding='latin1')


# Assuming your CSV has a column 'text' for SMS messages and a column 'label' for spam/legitimate
# If your columns have different names, replace 'text' and 'label' with the actual column names

# Extract 'text' and 'label' columns
X = df['v1'].tolist()
y = df['v2'].tolist()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v1'], df['v2'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Naive Bayes Classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Predictions
y_pred = nb_classifier.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

