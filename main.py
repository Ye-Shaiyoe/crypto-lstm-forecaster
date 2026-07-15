import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 1. Membuat data simulasi harga kripto (100 hari)
np.random.seed(42)
harga_asli = np.sin(np.linspace(0, 20, 100)) + np.random.normal(0, 0.1, 100)
harga_asli = harga_asli.reshape(-1, 1)

# 2. Normalisasi data ke skala 0 sampai 1
scaler = MinMaxScaler(feature_range=(0, 1))
harga_skala = scaler.fit_transform(harga_asli)

# 3. Menyiapkan data untuk LSTM (Gunakan 10 hari terakhir untuk prediksi hari ke-11)
X, y = [], []
window_size = 10

for i in range(len(harga_skala) - window_size):
    X.append(harga_skala[i : i + window_size])
    y.append(harga_skala[i + window_size])

X, y = np.array(X), np.array(y)

# 4. Membuat Arsitektur Model LSTM
model = Sequential([
    # Layer LSTM menerima input berbentuk [jumlah_sampel, window_size, fitur]
    LSTM(units=50, activation='relu', input_shape=(window_size, 1)),
    # Layer Dense untuk menghasilkan 1 angka prediksi output
    Dense(units=1)
])

# 5. Kompilasi dan Pelatihan Model
model.compile(optimizer='adam', loss='mean_squared_error')
print("Sedang melatih model...")
model.fit(X, y, epochs=20, batch_size=16, verbose=1)

# 6. Simulasi Prediksi untuk Hari Esok
# Mengambil 10 data terakhir dari pasar untuk memprediksi hari berikutnya
data_terakhir = harga_skala[-window_size:].reshape(1, window_size, 1)
prediksi_skala = model.predict(data_terakhir)

# Mengembalikan angka skala (0-1) ke nominal harga asli
prediksi_harga = scaler.inverse_transform(prediksi_skala)
print(f"\nHasil Prediksi Harga Kripto Besok: {prediksi_harga[0][0]:.4f}")
