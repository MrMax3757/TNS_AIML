from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 1. PROJECT PATHS

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "dataset" / "diabetes.csv"
MODEL_PATH = ROOT / "models"
REPORT_PATH = ROOT / "reports"

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)


# 2. LOAD DATASET

print("\n========== LOADING DATASET ==========")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())


# =====================================
# 3. DATA VISUALIZATION
# =====================================

# Create reports folder
figure_path = ROOT / "reports"
figure_path.mkdir(exist_ok=True)


# 1. TARGET DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Diabetes_binary"
)

plt.title("Diabetes Class Distribution")
plt.xlabel("Diabetes Status")
plt.ylabel("Number of Records")
plt.xticks([0, 1], ["No Diabetes", "Diabetes"])

plt.tight_layout()
plt.savefig(figure_path / "target_distribution.png")
plt.show()


# 2. BMI DISTRIBUTION

plt.figure(figsize=(8, 5))

sns.histplot(
    data=df,
    x="BMI",
    bins=30,
    kde=True
)

plt.title("BMI Distribution")
plt.xlabel("Body Mass Index")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(figure_path / "bmi_distribution.png")
plt.show()


# 3. AGE DISTRIBUTION

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="Age",
    hue="Diabetes_binary"
)

plt.title("Age Distribution by Diabetes Status")
plt.xlabel("Age Category")
plt.ylabel("Number of Records")

plt.legend(
    title="Diabetes Status",
    labels=["No Diabetes", "Diabetes"]
)

plt.tight_layout()
plt.savefig(figure_path / "age_distribution.png")
plt.show()


# 4. CORRELATION HEATMAP

plt.figure(figsize=(16, 12))

correlation_matrix = df.corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.3
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig(figure_path / "correlation_heatmap.png")
plt.show()

print("\nAll visualizations generated successfully!")

# =====================================
# 4. SPLIT DATASET
# =====================================

X = df.drop("Diabetes_binary", axis=1)
y = df["Diabetes_binary"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# =====================================
# 5. TRAIN RANDOM FOREST MODEL
# =====================================

print("\n========== MODEL TRAINING ==========")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed!")


# =====================================
# 6. MODEL EVALUATION
# =====================================

y_pred = model.predict(X_test)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# =====================================
# 7. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(REPORT_PATH / "confusion_matrix.png")
plt.show()

print("Confusion matrix saved!")


# =====================================
# 8. SAVE MODEL
# =====================================

joblib.dump(model, MODEL_PATH / "diabetes_model.pkl")

print("\nModel saved successfully!")


# =====================================
# 9. SAMPLE PREDICTION
# =====================================

patient = {
    "HighBP": 1,
    "HighChol": 1,
    "CholCheck": 1,
    "BMI": 28,
    "Smoker": 0,
    "Stroke": 0,
    "HeartDiseaseorAttack": 0,
    "PhysActivity": 1,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": 3,
    "MentHlth": 5,
    "PhysHlth": 5,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": 8,
    "Education": 5,
    "Income": 6
}

input_data = pd.DataFrame([patient])
input_data = input_data[X.columns]

prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0]

print("\n========== SAMPLE PREDICTION ==========")

if prediction == 1:
    print("Prediction: Diabetes-positive category")
else:
    print("Prediction: Diabetes-negative category")

print("Negative Probability:", round(probability[0] * 100, 2), "%")
print("Positive Probability:", round(probability[1] * 100, 2), "%")

print("\n========== PROJECT COMPLETED ==========")
