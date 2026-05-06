import pandas as pd

# CSV dosyasını oku
df = pd.read_csv("chd2_datas.csv")

# Son sütun = label
X = df.iloc[:, :-1]   # tüm sütunlar (son hariç)
y = df.iloc[:, -1]    # sadece son sütun

# CSV olarak kaydet
X.to_csv("chd2_data.csv", index=False)
y.to_csv("chd2_label.csv", index=False)

print("Data ve label başarıyla ayrıldı.")