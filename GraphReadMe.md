1. Accuracy Grafiği

Dosya örneği:

top_40_classifier_accuracy.png
Görevi: Modelin eğitim boyunca accuracy değerinin epoch epoch nasıl değiştiğini gösterir.

Yani:

x ekseni: epoch
y ekseni: accuracy
Bu grafik şunu cevaplar:

Model eğitim ilerledikçe daha doğru tahmin yapmayı öğreniyor mu?

Accuracy düzenli artıyorsa model öğreniyor demektir. Çok dalgalanıyorsa eğitim kararsız olabilir.

2. Convergence of Error Grafiği

Dosya örneği:

top_40_classifier_loss.png
Görevi: Modelin hata değerinin epoch boyunca nasıl değiştiğini gösterir.

Yani:

x ekseni: epoch
y ekseni: loss
Bu grafik şunu cevaplar:

Eğitim ilerledikçe modelin hatası azalıyor mu?

Loss düşüyorsa model daha iyi öğreniyor demektir. Loss düşmüyor veya artıyorsa model yeterince öğrenmiyor olabilir.

3. Average Accuracy Convergence Grafiği

Dosya örneği:

top_40_average_accuracy_convergence.png
Görevi: 50 tekrarın epoch bazlı ortalama accuracy değerini gösterir.

Yani her run için:

run_1 epoch accuracy
run_2 epoch accuracy
...
run_50 epoch accuracy
alınır, sonra her epoch için ortalama hesaplanır.

Bu grafik şunu cevaplar:

Model genel olarak, tekrarlar ortalamasında, eğitim boyunca nasıl öğreniyor?

Tek bir run şansa bağlı olabilir. Bu grafik daha güvenilir genel öğrenme eğrisidir.

4. Average Error Convergence Grafiği

Dosya örneği:

top_40_average_error_convergence.png
Görevi: 50 tekrarın epoch bazlı ortalama loss değerini gösterir.

Bu grafik şunu cevaplar:

Modelin ortalama hatası tekrarlar boyunca epoch ilerledikçe azalıyor mu?

Makale için önemli çünkü tek eğitim değil, çoklu çalıştırma ortalamasına göre yakınsamayı gösterir.

Boxplot Grafikleri

Boxplot grafikleri, modelin tekrar eden çalıştırmalarda ne kadar kararlı davrandığını göstermek için kullanılır. Senin kodunda boxplot’lar --repeat-runs ile yapılan çoklu çalıştırmaların sonuçlarından üretilir.

1. Accuracy Boxplot

Dosya örneği:

top_40_accuracy_boxplot.png
Bu grafik, 50 tekrar sonunda elde edilen final accuracy değerlerinin dağılımını gösterir.

Yani:

run_1 final accuracy
run_2 final accuracy
...
run_50 final accuracy
değerleri kullanılır.

Bu grafik şu soruya cevap verir:

Model farklı çalıştırmalarda benzer başarı değerleri veriyor mu?

Dar bir boxplot, accuracy değerlerinin birbirine yakın olduğunu ve modelin kararlı çalıştığını gösterir. Geniş bir boxplot ise model başarısının çalıştırmadan çalıştırmaya değiştiğini gösterir.

2. Loss Boxplot

Dosya örneği:

top_40_loss_boxplot.png
Bu grafik, 50 tekrar sonunda her çalıştırmanın son epoch’undaki final loss değerlerinin dağılımını gösterir.

Yani:

run_1 final loss
run_2 final loss
...
run_50 final loss
değerleri kullanılır.

Bu grafik şu soruya cevap verir:

Model farklı çalıştırmalarda benzer hata seviyesine yakınsıyor mu?

Düşük ve dar bir loss boxplot, modelin tekrarlar boyunca benzer ve düşük hata seviyesine ulaştığını gösterir. Geniş bir loss boxplot, bazı çalıştırmalarda modelin daha yüksek hata ile sonuçlandığını ve eğitimin kararlılığının daha zayıf olabileceğini gösterir.

Kısaca

Accuracy boxplot
→ 50 tekrar sonunda başarı değerleri kararlı mı? 

Loss boxplot
→ 50 tekrar sonunda hata değerleri kararlı mı? (Son loss değerleri alınır)
Bu iki grafik birlikte kullanıldığında modelin hem performans hem de hata açısından tekrar edilebilirliğini gösterir.

6. Clustering Silhouette Boxplot

Clustering için dosya örneği:

top_40_silhouette_boxplot.png
Görevi: Birden fazla clustering çalıştırmasında silhouette score dağılımını gösterir.

Bu grafik şunu cevaplar:

Clustering sonucu farklı çalıştırmalarda kararlı mı?

Silhouette yüksek ve dağılım dar ise clustering daha güvenilirdir. Düşük ve dağınıksa veri doğal kümelere iyi ayrılmıyor olabilir.


ÖZET : 

Accuracy grafiği
→ Tek eğitimde model öğreniyor mu?

Loss grafiği
→ Tek eğitimde hata azalıyor mu?

Average accuracy convergence
→ Çoklu tekrarların ortalama öğrenme davranışı nasıl?

Average error convergence
→ Çoklu tekrarların ortalama hata azalması nasıl?

Boxplot(Accuracy-Loss)
→ 50 tekrar sonunda başarı değerleri kararlı mı?
→ 50 tekrar sonunda hata değerleri kararlı mı? 