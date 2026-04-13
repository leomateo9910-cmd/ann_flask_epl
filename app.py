from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd

app = Flask(__name__)

model = load_model("model_epl.h5")
scaler = joblib.load("scaler.save")

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

# Fungsi grafik
def generate_chart(data):
    labels = ['Win', 'Draw', 'Lose', 'Goals', 'Points']

    plt.figure()
    plt.bar(labels, data)

    if not os.path.exists("static/chart"):
        os.makedirs("static/chart")

    plt.savefig("static/chart/chart.png")
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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
    df.to_excel("static/hasil_prediksi.xlsx", index=False)

    return render_template('index.html',
                           prediction=hasil,
                           team=team,
                           logo=logo,
                           data=data)

if __name__ == "__main__":
    app.run(debug=True)