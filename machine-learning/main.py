import os
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="Sleep Disorder Classification ML Service",
    description="Microservice prediksi gangguan tidur menggunakan Random Forest & SHAP",
    version="1.0.0",
)

# 2. Memuat model dan komponen preprocessing yang sudah dilatih
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_artifacts.joblib")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("File model_artifacts.joblib tidak ditemukan. Jalankan train.py terlebih dahulu.")

artifacts = joblib.load(MODEL_PATH)
rf_model = artifacts["model"]
preprocessor = artifacts["preprocessor"]
feature_names = artifacts["feature_names"]
classes = artifacts["classes"]

# Inisialisasi SHAP TreeExplainer untuk menjelaskan hasil prediksi Random Forest
explainer = shap.TreeExplainer(rf_model)


# 3. Skema data input yang diterima dari Backend Go
class SleepInput(BaseModel):
    gender: str = Field(..., example="Male")
    age: int = Field(..., ge=1, le=120, example=38)
    occupation: str = Field(..., example="Software Engineer")
    sleep_duration: float = Field(..., ge=0.0, le=24.0, example=6.1)
    quality_of_sleep: int = Field(..., ge=1, le=10, example=5)
    physical_activity_level: int = Field(..., ge=0, le=120, example=40)
    stress_level: int = Field(..., ge=1, le=10, example=8)
    bmi_category: str = Field(..., example="Overweight")
    systolic_bp: int = Field(..., ge=50, le=250, example=135)
    diastolic_bp: int = Field(..., ge=30, le=150, example=88)
    heart_rate: int = Field(..., ge=30, le=220, example=80)
    daily_steps: int = Field(..., ge=0, le=50000, example=4500)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ml-sleep-disorder", "classes": classes}


@app.post("/predict")
def predict_sleep_disorder(data: SleepInput):
    try:
        # Menyeragamkan kategori BMI seperti saat pelatihan model
        bmi_clean = "Normal" if data.bmi_category == "Normal Weight" else data.bmi_category

        # Susun input menjadi DataFrame sesuai nama kolom dataset
        input_df = pd.DataFrame([{
            "Gender": data.gender,
            "Age": data.age,
            "Occupation": data.occupation,
            "Sleep Duration": data.sleep_duration,
            "Quality of Sleep": data.quality_of_sleep,
            "Physical Activity Level": data.physical_activity_level,
            "Stress Level": data.stress_level,
            "BMI Category": bmi_clean,
            "Heart Rate": data.heart_rate,
            "Daily Steps": data.daily_steps,
            "Systolic_BP": data.systolic_bp,
            "Diastolic_BP": data.diastolic_bp,
        }])

        # Transformasi fitur menggunakan preprocessor tersimpan
        X_processed = preprocessor.transform(input_df)

        # Prediksi kelas dan probabilitas tiap kelas
        probabilities = rf_model.predict_proba(X_processed)[0]
        pred_idx = int(np.argmax(probabilities))
        predicted_class = classes[pred_idx]
        confidence = float(probabilities[pred_idx])

        prob_dict = {
            classes[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(classes))
        }

        # Hitung kontribusi tiap fitur menggunakan SHAP untuk kelas yang terprediksi
        shap_vals = explainer.shap_values(X_processed)
        if isinstance(shap_vals, list):
            class_shap = np.array(shap_vals[pred_idx][0])
        elif shap_vals.ndim == 3:
            class_shap = np.array(shap_vals[0, :, pred_idx])
        else:
            class_shap = np.array(shap_vals[0])

        # Urutkan 5 faktor paling berpengaruh (berdasarkan nilai absolut SHAP terbesar)
        shap_contributions = []
        for idx, feat_name in enumerate(feature_names):
            val = float(class_shap[idx])
            shap_contributions.append({
                "feature": feat_name,
                "shap_value": round(val, 4),
                "impact": "Meningkatkan risiko" if val > 0 else "Menurunkan risiko",
            })

        shap_contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        top_factors = shap_contributions[:5]

        return {
            "prediction": predicted_class,
            "confidence_percent": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "shap_explanations": top_factors,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses prediksi: {str(e)}")