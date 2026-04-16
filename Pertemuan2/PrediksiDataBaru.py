import pandas as pd
from MembangunModel import model, scaler

data_baru = pd.DataFrame({
    'luas_tanah': [120, 80],
    'jumlah_kamar': [3, 2],
    'jarak_ke_pusat_kota': [5, 15],
    'tahun_dibangun': [2010, 2000]
})

data_baru_processed = scaler.transform(data_baru)
prediksi_harga = model.predict(data_baru_processed)

print("Prediksi Harga Rumah (juta Rp):", prediksi_harga.round(2))