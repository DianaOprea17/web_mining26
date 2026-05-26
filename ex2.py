# [Example: Text Classification using 20 Newsgroups dataset]

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load a subset of the 20 Newsgroups dataset
#    We use only two categories:
#    - sci.space
#    - rec.sport.hockey

categories = ['sci.space', 'rec.sport.hockey']

newsgroups = fetch_20newsgroups(
    subset='all',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# 2. Extract text data (X) and labels (y)
X = newsgroups.data
y = newsgroups.target

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 4. Convert text into numerical features
#    CountVectorizer transforms words into numbers
vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectorized, y_train)

# 6. Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# 7. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.2f}")

# Optional: Show category names
print("\nCategories:")
print(newsgroups.target_names)