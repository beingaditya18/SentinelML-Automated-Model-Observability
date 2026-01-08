import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load data
df = pd.read_csv("data/raw/adult.csv")

df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

X = df.drop("income", axis=1)
y = df["income"]

cat_cols = X.select_dtypes(include="object").columns
num_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("✅ Model accuracy:", acc)

# Save artifacts
os.makedirs("model", exist_ok=True)
os.makedirs("data/reference", exist_ok=True)

joblib.dump(pipeline, "model/trained_model.pkl")
joblib.dump(preprocessor, "model/encoder.pkl")

X_train.assign(target=y_train).to_csv(
    "data/reference/reference_data.csv", index=False
)

print("✅ Model + encoder + reference data saved")
