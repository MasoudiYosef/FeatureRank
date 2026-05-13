1. Ortak Başlangıç
Komut çalışınca önce şunlar oluyor:

main()
GPU/CPU seçiliyor.
Random seed ayarlanıyor.
feature_percent kontrol ediliyor.
Dataset adı .txt ise .csvye çevriliyor.
Veri load_data() ile okunuyor.
Target kolonundaki sınıf sayısına bakılıyor:
class_count = df[target_column].nunique()
Eğer:

class_count <= 2 -> binary akış
class_count > 2  -> multi-class one-vs-rest akış
2. Binary Veri Nasıl Çalışıyor?
Binary veri için çalışan fonksiyon:

run_binary_experiment()
Akış:

preprocess_data(..., scale_features=False) çağrılıyor.
ID kolonu varsa atılıyor.
Target sayısala çevriliyor.
X ve y ayrılıyor.
Sayısal olmayan feature’lar temizleniyor.
Train/test split yapılıyor.
Bu aşamada tüm veri hemen scale edilmiyor. Bu iyi, çünkü büyük veri varsa gereksiz RAM yemiyor.
Feature sayısına bakılıyor.
Eğer feature sayısı 20000 veya altındaysa normal binary akış:

Tüm feature'larla autoencoder eğit
-> ORG accuracy hesapla
-> ağırlık listesini çıkar
-> top %feature_percent feature seç
-> seçilmiş feature datasetini oluştur
-> seçilmiş feature'larla tekrar eğit
-> filtered accuracy hesapla
Binary küçük/orta veri çıktıları:

outputs/autoencoder/{dataset}/first_layer_W_list.csv
outputs/autoencoder/{dataset}/top_20_max_abs_features.csv
data/autoencoder/{dataset}/top_20_max_abs_features_dataset.csv
outputs/autoencoder/{dataset}/metrics/ORG_test_metrics.json
outputs/autoencoder/{dataset}/metrics/top_20_test_metrics.json
Burada iki accuracy var:

ORG_test_metrics.json        -> tüm feature accuracy
top_20_test_metrics.json     -> seçilmiş feature accuracy
3. Büyük Boyutlu Veri Nasıl Çalışıyor?
Büyük veri kontrolü şu:

DEFAULT_CHUNK_FEATURE_THRESHOLD = 20000
DEFAULT_FEATURE_CHUNK_SIZE = 1000
Yani:

feature sayısı <= 20000 -> normal/global akış
feature sayısı > 20000  -> parçalı akış
Örnek:

arcene_data ≈ 10000 feature -> parçalanmaz
dorothea_data = 100000 feature -> parçalanır
Büyük veri için çalışan fonksiyon:

run_chunked_binary_experiment()
Parçalama satırdan değil, feature kolonundan yapılıyor.

Örneğin Dorothea:

X_train: 1035 x 100000
Şuna bölünüyor:

chunk_001 -> feature_1 ... feature_1000
chunk_002 -> feature_1001 ... feature_2000
...
chunk_100 -> feature_99001 ... feature_100000
Her chunk içinde bütün satırlar kalıyor, sadece feature sayısı azalıyor.

Her chunk için:

chunk verisini scale et
-> autoencoder eğit
-> classifier accuracy hesapla
-> first_layer_W_list.csv üret
-> chunk içindeki top %20 feature'ı seç
Sonra:

tüm chunk top %20 feature'ları birleştiriliyor
-> birleşik feature setiyle final eğitim yapılıyor
-> final accuracy hesaplanıyor
Büyük veri modunda önemli nokta:

100000 feature ile ORG accuracy üretilmiyor.
Çünkü zaten hata alınan kısım orasıydı. Bu modda ORG yerine final olarak şu var:

chunklardan seçilen feature'ların birleşimiyle final accuracy
Büyük veri çıktıları:

outputs/autoencoder/{dataset}/chunks/chunk_001/
outputs/autoencoder/{dataset}/chunks/chunk_002/
...
Her chunk içinde:

first_layer_W_list.csv
top_20_max_abs_features.csv
Final dosyalar:

outputs/autoencoder/{dataset}/chunked_top_20_max_abs_features.csv
outputs/autoencoder/{dataset}/chunked_merged_top_20_features.csv
data/autoencoder/{dataset}/chunked_top_20_max_abs_features_dataset.csv
outputs/autoencoder/{dataset}/metrics/chunked_top_20_test_metrics.json
outputs/autoencoder/{dataset}/metrics/top_20_test_metrics.json
4. Multi-Class Veri Nasıl Çalışıyor?
Multi-class için çalışan fonksiyon:

run_multiclass_one_vs_rest()
Kod gerçek multi-class softmax eğitimi yapmıyor. Onun yerine one-vs-rest yapıyor.

Örneğin sınıflar:

A, B, C
Kod üç ayrı binary problem oluşturuyor:

A vs rest
B vs rest
C vs rest
Ama senin kodunda özel olarak şu mantık var:

binary_df[target_column] = (binary_df[target_column] != class_label).astype(np.int32)
Yani seçilen class:

seçili class -> 0
diğer tüm class'lar -> 1
Her class için run_binary_experiment() çalışıyor. Dolayısıyla multi-class içinde de aynı kurallar geçerli:

feature <= 20000 ise normal binary akış
feature > 20000 ise chunked binary akış
Her class için ayrı klasör oluşuyor:

outputs/autoencoder/{dataset}/{class_label}_{dataset}/
Sonra her class’ın accuracy değeri okunuyor ve weighted average hesaplanıyor:

compute_multiclass_macro_accuracy()
İsmi macro_accuracy gibi ama kod aslında class count varsa weighted average yapıyor.

Multi-class final çıktıları:

outputs/autoencoder/{dataset}/metrics/ORG_test_metrics.json
outputs/autoencoder/{dataset}/metrics/top_20_test_metrics.json
5. Dorothea Özel Durumu
Dorothea normal CSV değil. Şöyle geliyor:

"191,367,614,..."
Yani her satır aktif feature index’lerini veriyor. Ben loader’a bunu ekledim:

_parse_sparse_index_feature_file()
Bu fonksiyon Dorothea’yı 0/1 matrise çeviriyor:

feature_1, feature_2, ..., feature_100000, target
Bu yüzden Dorothea artık şöyle okunuyor:

100000 feature + target
Ayrıca label dosyası data’dan 1 satır fazlaysa son label satırı kırpılıyor.

Kısa Özet
Binary küçük/orta veri:

Tüm feature -> accuracy
Top % feature -> tekrar accuracy
Multi-class veri:

Her sınıf için binary one-vs-rest
Sonra class accuracy'lerinden weighted average
Büyük feature veri:

Feature kolonlarını 1000'lik chunklara böl
Her chunk'tan top %20 seç
Seçilenleri birleştir
Final eğitim yap
Şu anki önemli eşik:

20000 feature üstü otomatik parçalanır.



