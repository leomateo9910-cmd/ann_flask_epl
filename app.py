from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib
matplotlib.use('Agg')  # 🔥 WAJIB (biar tidak crash)
import matplotlib.pyplot as plt
import os
import pandas as pd

app = Flask(__name__)

# 🔥 PATH AMAN
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model_epl.h5")
scaler_path = os.path.join(BASE_DIR, "scaler.save")

# 🔥 LOAD MODEL (sekali saja, aman)
model = load_model(model_path)
scaler = joblib.load(scaler_path)

# Logo mapping
logo_dict = {
    "Arsenal": "arsenal.png",
    "Manchester City": "mancity.png",
    "Liverpool": "liverpool.png",
    "Chelsea": "chelsea.png",
    "Manchester United": "mu.png",
    "Tottenham": "tottenham.png",
    "Aston Villa": "villa.png"
}

# Grafik
def generate_chart(data):
    labels = ['Win', 'Draw', 'Lose', 'Goals', 'Points']

    plt.figure()
    plt.bar(labels, data)

    chart_dir = os.path.join(BASE_DIR, "static/chart")
    if not os.path.exists(chart_dir):
        os.makedirs(chart_dir)

    chart_path = os.path.join(chart_dir, "chart.png")
    plt.savefig(chart_path)
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        team = request.form['team']

        data = [
            float(request.form['win']),
            float(request.form['draw']),
            float(request.form['lose']),
            float(request.form['goals']),
            float(request.form['points'])
        ]

        # Prediksi
        data_scaled = scaler.transform([data])
        pred = model.predict(data_scaled)

        hasil = "MASUK TOP 5 🔥" if pred[0][0] > 0.5 else "TIDAK MASUK TOP 5 ❌"
        logo = logo_dict.get(team, "default.png")

        # Grafik
        generate_chart(data)

        # Export Excel
        df = pd.DataFrame([{
            "Team": team,
            "Win": data[0],
            "Draw": data[1],
            "Lose": data[2],
            "Goals": data[3],
            "Points": data[4],
            "Prediction": hasil
        }])

        excel_path = os.path.join(BASE_DIR, "static/hasil_prediksi.xlsx")
        df.to_excel(excel_path, index=False)

        return render_template('index.html',
                               prediction=hasil,
                               team=team,
                               logo=logo,
                               data=data)

    except Exception as e:
        return f"ERROR: {str(e)}"

# 🔥 WAJIB UNTUK RAILWAY
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
