import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# 1) Load data
data = pd.read_csv("churn.csv")

# 2) Clean numeric column
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
data = data.dropna(subset=["TotalCharges"])

# 3) Target to numeric
data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

# 4) Drop ID column (not useful for prediction)
data = data.drop(columns=["customerID"])

# 5) Convert text columns to numbers (One-Hot Encoding)
data_encoded = pd.get_dummies(data, drop_first=True)

# 6) Split into features (X) and target (y)
X = data_encoded.drop("Churn", axis=1)
y = data_encoded["Churn"]

# 7) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8) Train a baseline model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# 9) Predict + evaluate
y_pred = model.predict(X_test)

print("✅ Model Trained Successfully")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nData shape (rows, cols):", data.shape)
print("Encoded shape (rows, cols):", data_encoded.shape)