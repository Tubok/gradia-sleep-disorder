import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 1. Menentukan lokasi folder agar path selalu akurat
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Sleep_health_expanded.csv")

print("Membaca dataset...")
df = pd.read_csv(csv_path)

# 2. Pembersihan Data (Data Cleaning)
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

bp_split = df["Blood Pressure"].str.split("/", expand=True)
df["Systolic_BP"] = pd.to_numeric(bp_split[0])
df["Diastolic_BP"] = pd.to_numeric(bp_split[1])

X = df.drop(columns=["Person ID", "Blood Pressure", "Sleep Disorder"])
y = df["Sleep Disorder"]

print("\nDistribusi data awal sebelum SMOTE:")
print(y.value_counts())

# 3. Stratified 80:20 Split sebelum SMOTE agar data uji tetap murni
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. One-Hot Encoding (kategorikal) + StandardScaler (numerik)
categorical_cols = ["Gender", "Occupation", "BMI Category"]
numeric_cols = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Systolic_BP",
    "Diastolic_BP",
    "Heart Rate",
    "Daily Steps",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_cols,
        ),
        ("num", StandardScaler(), numeric_cols),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = [
    name.replace("cat__", "").replace("num__", "")
    for name in preprocessor.get_feature_names_out()
]

# 5. ImbPipeline (SMOTE + Random Forest) & Tuning GridSearchCV (5-Fold CV)
pipeline_rf = ImbPipeline(
    [
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(random_state=42)),
    ]
)

param_grid = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth": [5, 10, 15, None],
    "rf__min_samples_split": [2, 4, 5],
    "rf__criterion": ["gini", "entropy"],
}

print("\nMenjalankan GridSearchCV (5-Fold Cross Validation)...")
grid_rf = GridSearchCV(
    estimator=pipeline_rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1,
)
grid_rf.fit(X_train_processed, y_train)

best_rf_model = grid_rf.best_estimator_.named_steps["rf"]
print(f"Parameter Terbaik: {grid_rf.best_params_}")

# 6. Evaluasi Performa Model Terbaik pada Data Uji Murni
y_pred = best_rf_model.predict(X_test_processed)
acc = accuracy_score(y_test, y_pred)

print(f"\n=== HASIL EVALUASI MODEL ===")
print(f"Akurasi pada Data Uji: {acc * 100:.2f}%\n")
print("Laporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 7. Menyimpan Model & Komponen Preprocessing untuk API FastAPI + SHAP
X_train_smote, _ = SMOTE(random_state=42).fit_resample(X_train_processed, y_train)

artifacts = {
    "model": best_rf_model,
    "preprocessor": preprocessor,
    "feature_names": feature_names,
    "classes": best_rf_model.classes_.tolist(),
    "background_data": X_train_smote[:100],
}

output_model_path = os.path.join(BASE_DIR, "model_artifacts.joblib")
joblib.dump(artifacts, output_model_path)
print(f"Model berhasil disimpan ke: {output_model_path}")