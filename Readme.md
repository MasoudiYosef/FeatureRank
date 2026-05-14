OTOMATIK ORTAM KURULUMU:
 - "bash scripts/setup_environment.sh" => Python 3.13.5 kontrol eder, pyenv varsa eksikse kurar, .venv oluşturur ve requirements.txt paketlerini yükler.
 - "source .venv/bin/activate" => Kurulumdan sonra sanal ortamı aktif eder.

MANUEL ORTAM KURULUMU:
 - "python3 --version" => Python sürümünü kontrol et. Bu proje için önerilen sürüm Python 3.13.5.
 - "python3 -m venv .venv" => Bu komut proje içinde .venv klasörü oluşturur.
 - "source .venv/bin/activate" => Başarılı olursa terminalin başında genelde şöyle bir şey görünür: (.venv)
 - "python --version" => Sanal ortam içinde de Python sürümünün doğru olduğunu kontrol et.

TENSORFLOW KURULUM HATASI:
 - "No matching distribution found for tensorflow==2.21.0" hatası genelde Python sürümü veya işletim sistemi/platform uyumsuzluğundan olur.
 - Manuel kurulumda "python3" komutu Python 3.13.5 değilse önce "bash scripts/setup_environment.sh" komutunu kullan.
 - Eski/yanlış .venv oluşturulduysa klasörü silip ortamı yeniden kur.

Kurulan Kütüphaneler :
 - "pip install -r requirements.txt" => CPU ile çalışmak için
 - "pip install -r requirements-gpu.txt" => GPU ile çalışmak için (bkz: GPU_SETUP.md)
 - "pip freeze > requirements.txt" => Projeyi başka bilgisayarda açarsan aynı kütüphaneleri tekrar kurabilirmemizi sağlar.

GPU KURULUMU:
 - GPU'lu bir bilgisayarda çalıştırmak için GPU_SETUP.md dosyasını oku
 - CUDA, cuDNN ve NVIDIA Driver kurulum talimatları içerir
 - requirements-gpu.txt ile uyumlu TensorFlow kurulması sağlanır


Tanımlar :
pandas → veri okuma ve tablo işlemleri
numpy → sayısal işlemler
scikit-learn → preprocessing, split, metrics
matplotlib → grafik
seaborn → veri görselleştirme
openpyxl → Excel dosyasını okumak için
tensorflow → CNN ve autoencoder modeli için


Amacımız : Bir veri setindeki feature’ların hangilerinin daha önemli olduğunu CNN veya Autoencoder kullanarak bulmak ve bunları sıralamak.


CNN Çalışma Sistemi :
    CNN’de feature importance → input feature → convolution → activation → output etkisi

    CNN, özellikler (feature’lar) arasındaki yerel ilişkileri ve etkileşimleri öğrenir.
    Birden fazla katman sayesinde hem tekil feature etkisini hem de feature’ların birlikte oluşturduğu örüntüleri (patterns) yakalar.

    Modeli eğitmek için çok katmanlı bir CNN yapısı kullandık. İlk katman feature’ları tek tek işlerken, sonraki katmanlar feature’lar arasındaki ilişkileri ve daha karmaşık pattern’leri öğrenir.

    Eğitim sırasında tüm katmanlar birlikte öğrenir, yani 1. katmanın ağırlıkları aslında 2. ve 3. katmanların etkisiyle şekillenir.

    Ancak feature ranking yaparken tüm katmanlara bakmayız. Çünkü derin katmanlarda feature’lar karışır ve yorumlanamaz hale gelir.

    Bu yüzden sadece ilk katmanın aktivasyonlarını kullanarak, hangi feature’ın daha güçlü sinyal ürettiğini ölçer ve buna göre sıralama yaparız.

    Tüm katmanlar öğrenme sürecine katkı verir, ancak feature importance doğrudan ilk katmandan çıkarılır.

    Bu projede önce veriyi yükleyip CNN modelini kurduk. Model, birden fazla convolution katmanı kullanarak sınıflandırma problemini öğrendi. Eğitim tamamlandıktan sonra feature importance’ı activations’tan değil, doğrudan ilk convolution katmanının kernel ağırlıklarından çıkardık. İlk conv katmanının kernel matrisi feature × filter biçiminde tabloya dönüştürüldü. Ardından her feature için filtreler arasındaki maksimum ve ortalama mutlak ağırlık hesaplandı. Bu skorlar kullanılarak feature’lar büyükten küçüğe sıralandı ve feature ranking dosyaları oluşturuldu.

Train Mantığı :
    Bu script iki farklı modda çalışıyor:

    original mode
        Orijinal veri setini yükler, preprocess eder, autoencoder eğitir, test metriğini hesaplar, encoder ağırlıklarından feature ranking çıkarır, sonra bu ranking’e göre filtrelenmiş yeni datasetler üretir.
    filtered mode
        Daha önce oluşturulmuş filtrelenmiş datasetlerden birini yükler, tekrar preprocess eder, autoencoder’ı o filtrelenmiş veri üzerinde çalıştırır ve test metriğini hesaplar.

    Yani mantık şu:

    1.Ham veriyle autoencoder eğit
    2.İlk encoder katmanının ağırlıklarına bak
    3.Hangi feature daha “güçlü” katkı veriyor diye sırala
    4.En iyi yüzde kaç feature isteniyorsa seç
    5.Yeni CSV üret
    



AutoEncoder Çalışma Mantığı :

    x→Encoder→z→Decoder→x^ (Mantığı)
    minimize ∣∣x−x^∣∣ (Amaç)

    Autoencoder, veriyi daha düşük boyutlu bir temsile (latent space) sıkıştırıp tekrar orijinaline yakın şekilde yeniden üretmeye çalışır.
    Bu süreçte model, veriyi en iyi temsil eden özellikleri öğrenir.
    
    Input: orijinal feature vektörü
    Encoder: veriyi daha küçük temsile indirir
    Bottleneck: sıkıştırılmış özet temsil
    Decoder: bu özeti kullanıp giriş veriyi tekrar üretmeye çalışır

    Örnek:
        Girişte 30 feature var
        Encoder bunu 16’ya indirir
        Sonra 8’e indirir
        Decoder tekrar 16’ya çıkarır
        Sonra yeniden 30 feature üretir

    Burada modelin amacı sınıf tahmini değil,input’u yeniden üretmek

FARKI (CNN - AUTOENCODER)
    CNN
        supervised
        sınıflandırma öğrenir
        label kullanır
        ranking, sınıf ayırmaya katkı açısından yorumlanabilir

    Autoencoder
        unsupervised
        yeniden üretim öğrenir
        label kullanmaz
        ranking, veri yapısını temsil etme açısından yorumlanabilir

Bu çok önemli. Çünkü Autoencoder ranking’i şu anlama gelir:

    “Bu feature verinin yapısını yeniden kurmak için ne kadar önemli?”

CNN ranking’i ise daha çok şunu söyler:

    “Bu feature sınıflandırma için ne kadar etkili?

K-means ile Cluster Ayarlama : 

    Feature’ları CNN’den elde edilen ağırlık temsillerine göre K-Means ile cluster’la.
    Sonra her cluster içindeki feature’lar arasından, importance skoru en yüksek olan tek bir feature seç.
    Böylece hem önemli hem de birbirinden farklı feature’lardan oluşan yeni bir feature set elde et.
    Weight-based representation + K-Means ile cluster bazlı temsilci feature seçimi:
    1.Her feature için Conv kernel temsili alınıyor.
    2.K-Means ile cluster’a ayrılıyor.
    3.Seçtiğiniz ranking tipine göre importance hesaplanıyor:
    4.max: cluster_select_representative_features içinde max abs kernel ağırlığı
    5.avg: cluster_select_representative_features içinde mean abs kernel ağırlığı
    6.Her cluster’dan en yüksek importance feature seçiliyor. (Exp. count = 3 3 cluster )


    AUTOENCODER: 
        1. Original :

        python scripts/run_autoencoder.py --dataset-name heart_disease_data.csv --target-column target --id-column none 

        2. Filtered :

        python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 10 --random-seed 42 

        3. 50 kere çalıştır :

        python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv --target-column diagnosis --id-column none --feature-percent 30 --random-state none --repeat-runs 50

        4. Task
          python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv --task clustering
          python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv --task classification
    
        5. Save History
           python scripts/run_autoencoder.py --dataset-name breast_cancer_data.csv --save-training-plots
     
    "feature_percent": 20.0,
    "selected_feature_count": 6,
    "test_mse": 0.5953885912895203,
    "test_accuracy": 0.9122807017543859,
    "threshold": 0.5

    Purpose: 
    1.Veriyi yüklemek ve ön işlemek
    2.Autoencoder eğitmek
    3.Encoder çıktılarıyla bir sınıflandırıcı eğitmek
    4.İlk katmandaki ağırlıklardan feature önemini çıkarmak
    5.En iyi feature’ları seçip yeni bir veri seti oluşturmak
    6.Bu filtrelenmiş veri setiyle aynı süreci tekrar çalıştırmak
    7.Eski ve yeni sonuçları dosyaya kaydetmek

Bu kod, önce veri setini yükleyip ön işleme tabi tutuyor, ardından bir autoencoder eğiterek verinin sıkıştırılmış bir temsilini öğreniyor; sonra encoder çıktıları üzerinde ayrı bir sınıflandırıcı kurup test MSE ve test accuracy değerlerini hesaplıyor. Eğitim tamamlandıktan sonra autoencoder’ın ilk encoder katmanındaki ağırlıkları ve eğitim verisini birlikte kullanarak her feature’ın gizli nöronlara ortalama katkı listesini çıkarıyor, bu katkılardan en güçlü yüzde kadar feature’ı seçip yeni bir filtrelenmiş veri seti oluşturuyor, ardından aynı eğitim ve değerlendirme sürecini bu küçültülmüş veri seti üzerinde tekrar çalıştırıyor. Son olarak hem orijinal veriyle hem de seçilen feature’larla elde edilen performans sonuçlarını JSON ve CSV dosyalarına kaydedip kullanıcıya raporluyor; yani kodun temel amacı autoencoder ile temsil öğrenmek, bu temsilden feature önemini çıkarmak ve feature selection sonrası performansın nasıl değiştiğini karşılaştırmak.

BİNARY MODEL:
    0-1 target değerlerinden oluşur ve buna göre işlem yapılır.

MULTİ CLASS MODEL:
    Eğer data multi class ise yani içinde 0-1 dışında daha fazla sınıf barındırıyorsa, her birini binary hale getiriyoruz. Örnek 4 class varsa her birini 0 yapıp diğerlerini 1 yapıp yeni bir label elde ediyoruz. bu örnek için 4 adet yeni label datası elde ediyoruz her birinde ayrı ayrı test ediyoruz. ve en son bir metot ile ortalamalarını alıp bitiricez bu modeli.

{
    "test_mse": 1.04411780834198,
    "test_accuracy": 0.8360655737704918,
    "threshold": 0.5,
    "current_class_label": 0,
    "class_counts": {
        "0": 164,
        "1": 55,
        "2": 36,
        "3": 35,
        "4": 13
    },
    "binary_label_counts": {
        "label_0": 33,  # Test (%20)
        "label_1": 28   # Test (%20)
    }
}


BÜYÜK VERİLERDE;
    100000 feature 
        -> 100 parçaya böl
        -> her 1000 feature için autoencoder çalıştır
        -> her parçadan en iyi %20 al (Ağırlık listesine bakarak)
        -> yaklaşık 20000 feature birleştir
        -> final autoencoder + classifier çalıştır
        -> final accuracy ver
