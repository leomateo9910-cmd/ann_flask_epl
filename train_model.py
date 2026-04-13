import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("data_epl_full.csv")

X = df[['Win','Draw','Lose','Goals','Points']]
y = df['Top5']

# Normalisasi
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model ANN
model = Sequential()
model.add(Dense(10, input_dim=5, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training
model.fit(X_train, y_train, epochs=100, verbose=1)

# Evaluasi
y_pred = (model.predict(X_test) > 0.5).astype(int)
print("Akurasi:", accuracy_score(y_test, y_pred))

# Save
model.save("model_epl.h5")
joblib.dump(scaler, "scaler.save")