import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("data.csv", header=None, low_memory=False)

# İlk satırı sil (header / özellik isimleri)
df = df.iloc[1:, :]

# İlk sütunu sil (örnek isimleri / ID)
df = df.iloc[:, 1:]

# En son sütunu (label) çıkar
data_only = df.iloc[:, :-1]

# Satır ve sütun sayılarını al
satir_sayisi = df.shape[0]
sutun_sayisi = df.shape[1]

# Yeni CSV olarak kaydet
data_only.to_csv("gen_data.csv", index=False, header=False)

print("Satır sayısı:", satir_sayisi)
print("Sütun sayısı:", sutun_sayisi)
print("Yeni boyut:", data_only.shape)