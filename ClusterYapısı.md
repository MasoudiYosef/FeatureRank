Bu projede clustering işlemi, label bilgisini kullanmadan verinin kendi içindeki doğal gruplarını bulmaya çalışır. Classification’dan farkı şudur: classification’da model gerçek sınıf etiketlerini öğrenir; clustering’de ise model sadece feature değerlerine bakarak örnekleri birbirine benzerliklerine göre kümelere ayırır.

Clustering Nedir?
Clustering, gözetimsiz öğrenme yöntemidir. Amaç, veri noktalarını birbirine benzer olanlar aynı grupta, birbirinden farklı olanlar farklı grupta olacak şekilde ayırmaktır.

Bu projede kullanılan yöntem:

KMeans clustering
KMeans şu mantıkla çalışır:

Kullanıcı bir k değeri verir.
k tane cluster merkezi oluşturulur.
Her örnek en yakın cluster merkezine atanır.
Her cluster merkezi, kendisine atanmış örneklerin ortalamasıyla güncellenir.
Bu işlem merkezler stabil hale gelene kadar devam eder.
KMeans’in amacı cluster içi uzaklığı minimize etmektir. Bu değer genelde inertia olarak raporlanır.

Bu Projede Clustering Akışı
Senin projende clustering şu şekilde çalışıyor:

Veri yüklenir
Label varsa ayrılır ama clustering eğitiminde kullanılmaz
Classification ile aynı autoencoder feature ranking kullanılır
Seçilen top % feature alınır
KMeans uygulanır
Silhouette score hesaplanır
Sonuç JSON/CSV olarak kaydedilir
Yani clustering modeli label’ı görmez. Label sadece class sayısını belirlemek için kullanılır:

k = class_count
Örneğin:

breast_cancer -> 2 class -> k=2
codon_usage -> 11 class -> k=11
cortex -> 8 class -> k=8
Bu kararın akademik karşılığı şudur: clustering sonucu gerçek sınıf yapısıyla karşılaştırılabilir hale getirilir, çünkü cluster sayısı bilinen sınıf sayısına eşitlenir.

Feature Selection ile İlişkisi
Bu projede clustering doğrudan tüm feature’larla yapılmaz. Önce autoencoder tabanlı feature ranking yapılır.

Autoencoder şu amaçla eğitilir:

Girdiyi sıkıştırıp tekrar üretmek
Autoencoder’ın ilk encoder katmanındaki ağırlıklar kullanılarak feature önem skorları çıkarılır. Daha sonra belirli yüzdelerde en yüksek ağırlığa sahip feature’lar seçilir:

top %20
top %30
top %50
...
Clustering bu seçilmiş feature subset’i üzerinde yapılır.

Bu sayede şu soru araştırılır:

Autoencoder ağırlıklarına göre seçilen feature’lar,
verinin doğal küme yapısını ne kadar iyi ortaya çıkarıyor?
Elbow Method Nedir?
Elbow method, KMeans için uygun cluster sayısını seçmekte kullanılan klasik bir yöntemdir.

KMeans her k değeri için inertia üretir.

Inertia:

Her örneğin kendi cluster merkezine olan uzaklıklarının kareleri toplamı
Matematiksel olarak:

Inertia = sum ||x_i - c_j||²
Burada:

x_i = veri noktası
c_j = x_i'nin ait olduğu cluster merkezi
Inertia ne kadar düşükse, cluster içindeki noktalar merkezlerine o kadar yakındır.

Ancak dikkat: k arttıkça inertia neredeyse her zaman düşer. Çünkü daha fazla cluster merkezi kullanıldığında noktaları merkeze yaklaştırmak kolaylaşır.

Örneğin:

k=2 -> inertia yüksek
k=3 -> daha düşük
k=4 -> daha düşük
...
Bu yüzden sadece en düşük inertia seçilmez. Elbow method’da amaç, inertia düşüşünün belirgin şekilde yavaşladığı noktayı bulmaktır.

Bu nokta “dirsek” olarak adlandırılır.

Örnek:

k=2 -> 1000
k=3 -> 600
k=4 -> 430
k=5 -> 390
k=6 -> 370
Burada k=4 sonrası düşüş yavaşlıyorsa, k=4 makul cluster sayısı olabilir.

Senin projende artık k label class sayısına eşitleniyor; yine de inertia raporlanıyor çünkü cluster içi sıkılığı yorumlamak için faydalıdır.

Silhouette Score Nedir?
Silhouette score, clustering kalitesini ölçmek için kullanılan en önemli metriklerden biridir.

Her veri noktası için iki değer hesaplanır:

a(i) = aynı cluster içindeki noktalara ortalama uzaklık
b(i) = en yakın farklı cluster içindeki noktalara ortalama uzaklık
Sonra silhouette değeri şöyle hesaplanır:

s(i) = (b(i) - a(i)) / max(a(i), b(i))
Yorum:

a(i) küçük olmalı -> kendi clusterına yakın
b(i) büyük olmalı -> diğer clusterlardan uzak
Silhouette score aralığı:

-1 ile +1
Yorum:

+1'e yakın -> çok iyi ayrışmış cluster
0 civarı   -> clusterlar iç içe / sınırlar belirsiz
-1'e yakın -> nokta yanlış clusterda olabilir
Pratik yorum aralığı:

0.00 - 0.25 -> zayıf cluster ayrımı
0.25 - 0.50 -> orta düzey ayrım
0.50 - 0.70 -> iyi ayrım
0.70+       -> çok güçlü ayrım
Örneğin:

silhouette = 0.47
orta-iyi bir cluster ayrımıdır.

silhouette = 0.16
zayıf cluster ayrımıdır.

Neden Classification Accuracy Yüksek, Silhouette Düşük Olabilir?
Bu çok önemli.

Classification ve clustering aynı şeyi ölçmez.

Classification:

Label bilgisini kullanır
Sınıf karar sınırı öğrenir
Accuracy hesaplar
Clustering:

Label kullanmaz
Sadece mesafelere bakar
Kompakt ve ayrık cluster arar
Silhouette hesaplar
Bir veri setinde sınıflar classification modeli tarafından çok iyi ayrılabilir ama KMeans açısından kompakt kümeler oluşturmayabilir.

Yani:

accuracy = 0.98
silhouette = 0.16
çelişki değildir.

Bu şu anlama gelir:

Feature’lar label tahmini için ayırt edici olabilir,
ama veriler mesafe uzayında doğal ve kompakt clusterlar oluşturmuyor olabilir.
Özellikle biyomedikal ve biyolojik verilerde bu çok sık görülür.

KMeans’in Sınırlılığı
KMeans bazı varsayımlar yapar:

Clusterlar yaklaşık küresel olmalı
Cluster yoğunlukları benzer olmalı
Euclidean distance anlamlı olmalı
Cluster boyutları çok dengesiz olmamalı
Eğer veri şu şekildeyse:

iç içe geçmiş sınıflar
uzun/karmaşık şekilli dağılımlar
dengesiz sınıflar
yüksek boyutlu sparse yapı
KMeans düşük silhouette verebilir.

Bu, veride bilgi olmadığı anlamına gelmez. Sadece KMeans’in bu yapıyı iyi yakalayamadığı anlamına gelir.

Bu Projedeki Akademik Yorum
Bu projede clustering deneyinin amacı şudur:

Autoencoder tabanlı feature selection ile seçilen feature’ların,
sadece supervised classification için değil,
unsupervised clustering için de anlamlı yapı taşıyıp taşımadığını incelemek.
Bu nedenle iki sonuç birlikte yorumlanır:

Classification accuracy
Silhouette score
Eğer accuracy yüksek, silhouette de yüksekse:

Seçilen feature’lar hem label tahmini hem doğal küme yapısı için güçlüdür.
Eğer accuracy yüksek, silhouette düşükse:

Seçilen feature’lar supervised karar sınırı için faydalıdır,
ancak label’sız mesafe tabanlı kümelenmede güçlü ayrışma üretmemektedir.
Eğer accuracy düşük, silhouette yüksekse:

Veride doğal kümeler vardır fakat bunlar verilen label’larla birebir örtüşmeyebilir.
Makaleye Yazılabilecek Özet
Şöyle yazabilirsin:

Clustering analysis was performed to evaluate whether the autoencoder-based selected features reveal an intrinsic grouping structure in the data independent of class labels. KMeans clustering was applied on the selected feature subsets, where the number of clusters was set equal to the number of known classes when labels were available. The silhouette score was used as the primary clustering validation metric, while inertia was reported to support elbow-based interpretation. A higher silhouette score indicates compact and well-separated clusters, whereas lower values suggest overlapping or weakly separated cluster structures.
Türkçe hali:

Clustering analizi, autoencoder tabanlı seçilen feature’ların sınıf etiketlerinden bağımsız olarak veride doğal bir grup yapısı ortaya çıkarıp çıkarmadığını değerlendirmek amacıyla uygulanmıştır. KMeans algoritması seçilen feature alt kümeleri üzerinde çalıştırılmış, label bilgisi mevcut olduğunda cluster sayısı gerçek sınıf sayısına eşitlenmiştir. Clustering başarımı temel olarak silhouette score ile değerlendirilmiş, inertia değeri ise elbow yöntemi kapsamında cluster içi sıkılığı yorumlamak için raporlanmıştır. Yüksek silhouette skoru kompakt ve iyi ayrılmış cluster yapısını, düşük silhouette skoru ise örtüşen veya zayıf ayrışmış cluster yapısını göstermektedir.
