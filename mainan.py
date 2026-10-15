import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ==========================================
# 1. LOAD MODEL, SCALER, DAN TRAINING COLUMNS
# ==========================================
MODEL_DIR = r"C:\Users\vb217249\OneDrive - PT.JFE Steel Galvanizing Indonesia\Training AI Engineer\PROJECT_AI\notebook"

MODEL_FILE = os.getenv("MODEL_FILE", os.path.join(MODEL_DIR, "linear_regression_model.pkl"))
SCALER_FILE = os.getenv("SCALER_FILE", os.path.join(MODEL_DIR, "scaler.pkl"))
COLUMNS_FILE = os.getenv("COLUMNS_FILE", os.path.join(MODEL_DIR, "training_columns.pkl"))

regmodel = joblib.load(MODEL_FILE)
scalar = joblib.load(SCALER_FILE)
training_columns = joblib.load(COLUMNS_FILE)

# ==========================================
# DEBUG: PERIKSA MODEL DAN SCALER
# ==========================================
print("\n========== MODEL DIAGNOSTIC ==========")
print("Model file:", MODEL_FILE)
print("Scaler file:", SCALER_FILE)
print("Columns file:", COLUMNS_FILE)
print("Jumlah training columns:", len(training_columns))
print("Daftar training columns:", training_columns)
print("======================================\n")

# ==========================================
# 2. FUNGSI PREDIKSI (SESUAI 7 KOLOM TRAINING)
# ==========================================
def predict_gas(form_data):
    thickness = float(form_data.get("THICKNESS", 0))
    width = float(form_data.get("WIDTH", 0))
    coil_weight = float(form_data.get("COIL_WEIGHT", 0)) * 1000.0  # Konversi Ton ke Kg (jika input web dalam Ton)
    coil_length = float(form_data.get("COIL_LENGTH", 0))
    furnace_code = float(form_data.get("FURNACE_CODE", 0))
    line_speed = float(form_data.get("LINE_SPEED", 0))

    # Hitung Thermal_Load otomatis sesuai rumus training
    thermal_load = width * thickness * line_speed

    # Petak-kan persis ke urutan 7 training_columns
    input_data = {
        'INLET_COIL_TH': thickness,
        'INLET_COIL_WH': width,
        'INLET_COIL_WT': coil_weight,
        'INLET_COIL_LG': coil_length,
        'NO1_ANLNG_CD': furnace_code,
        'AVE_CE_SP': line_speed,
        'Thermal_Load': thermal_load
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[training_columns]

    print("\n[DEBUG] Nilai input DataFrame:\n", input_df)
    
    scaled_input = scalar.transform(input_df)
    print("[DEBUG] Nilai setelah scaler.transform:", scaled_input)

    prediction = regmodel.predict(scaled_input)[0]

    return float(prediction)


# ==========================================
# 3. HALAMAN UTAMA
# ==========================================
@app.route("/")
def home():
    return render_template("home.html")


# ==========================================
# 4. PREDIKSI DARI FORM HTML
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        thickness = request.form.get("THICKNESS", "")
        width = request.form.get("WIDTH", "")
        coil_weight = request.form.get("COIL_WEIGHT", "")
        coil_length = request.form.get("COIL_LENGTH", "")
        furnace_code = request.form.get("FURNACE_CODE", "")
        line_speed = request.form.get("LINE_SPEED", "")

        output = predict_gas(request.form)

        return render_template(
            "home.html",
            prediction_text=f"Prediksi Konsumsi Natural Gas: {output:.2f}",
            thickness=thickness,
            width=width,
            coil_weight=coil_weight,
            coil_length=coil_length,
            furnace_code=furnace_code,
            line_speed=line_speed
        )

    except Exception as e:
        app.logger.exception("Prediksi gagal")
        return render_template(
            "home.html",
            prediction_text=f"Prediksi gagal: {str(e)}"
        ), 400


# ==========================================
# 5. HEALTH CHECK
# ==========================================
@app.route("/health")
def health():
    return "OK", 200


# ==========================================
# 6. JALANKAN SERVER
# ==========================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)