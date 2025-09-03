from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("churn-dataset-medium.csv")

# Separate categorical and numeric features
numeric_features = ["age", "daily_watch_hours", "tenure", "last_login", "promotions_used"]
categorical_features = ["gender", "subscription_type", "profiles", "genre_preference", "region"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=True), categorical_features)
    ],
    sparse_threshold=0.1  # ensure sparse output
)

# Build pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(C=0.1, penalty="l1", max_iter=1000, solver="saga", class_weight="balanced"))
])

# Train/test split
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print("Classification Report - Logistic Regression:")
print(classification_report(y_test, y_pred))
