import pandas as pd

df = pd.read_csv("gen_datas.csv",low_memory=False)

df = df.iloc[1:].reset_index(drop=True)

if df.shape[1] > 1:
    first_col = df.columns[0]
    if str(first_col).lower() in ["id", "index", "no", "sample_id"]:
        df = df.iloc[:, 1:]

label = df.iloc[:, -1]

data = df.iloc[:, :-1]

data.to_csv("gen_data.csv", index=False, header=False)
label.to_csv("gen_label.csv", index=False, header=False)

print("Header tamamen silindi")
print("Label ayrıldı")