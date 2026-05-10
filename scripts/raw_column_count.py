import pandas as pd

# # CSV dosyasını oku (sorunlu satırları atla)
# df = pd.read_csv("dorothea_data.csv", header=None, low_memory=False, on_bad_lines='skip')
# # İlk satırı sil (header / özellik isimleri)
# #df = df.iloc[1:, :]

# # İlk sütunu sil (örnek isimleri / ID)
# #df = df.iloc[:, 1:]

# # En son sütunu (label) çıkar
# #data_only = df.iloc[:, :-1]

# # Satır ve sütun sayılarını al
# satir_sayisi = df.shape[0]
# sutun_sayisi = df.shape[1]

# # Yeni CSV olarak kaydet
# #df.to_csv("dorothea_data.csv", index=False, header=False)

# print("Satır sayısı:", satir_sayisi)
# print("Sütun sayısı:", sutun_sayisi)
# print("Yeni boyut:", df.shape)



dosya_adi = "dorothea_data.csv"

satir_sayisi = 0
max_feature_index = 0
toplam_aktif_deger = 0

with open(dosya_adi, "r", encoding="utf-8") as f:
    for satir in f:
        satir = satir.strip()

        if satir == "":
            continue

        satir_sayisi += 1

        # Virgül ve boşlukları ayır
        parcalar = satir.replace(",", " ").split()

        # Float -> int dönüşümü
        sayilar = [int(float(x)) for x in parcalar]

        toplam_aktif_deger += len(sayilar)

        if sayilar:
            satir_max = max(sayilar)

            if satir_max > max_feature_index:
                max_feature_index = satir_max

print("Satır sayısı:", satir_sayisi)
print("Gerçek feature sayısı:", max_feature_index)
print("Toplam aktif değer:", toplam_aktif_deger)
print("Ortalama aktif feature:", toplam_aktif_deger / satir_sayisi)