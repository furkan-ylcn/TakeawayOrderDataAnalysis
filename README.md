# Restoran Sipariş Verisi Analizi

## Genel Bakış
Bu depo, bir restoran satış veri setine dayanarak çeşitli analizler ve tahminler yapmak için tasarlanmış üç Python dosyası içermektedir. Her dosya, belirli metrikleri analiz etmek ve tahmin etmek için makine öğrenimi modelleri kullanır. Aşağıda her bir dosyanın, işlevselliği ve sonuçlarının analizi hakkında ayrıntılı bir açıklama yer almaktadır.

---

## 1. **UrunSayisiTahmini.py**
### Amaç:
Bu dosya, toplam sipariş değeri temelinde bir siparişteki ürün sayısını Doğrusal Regresyon kullanarak tahmin eder.

### İş Akışı:
1. `restaurant_sales_data.csv` dosyasından verileri okur.
2. Eksik değer içeren satırları temizler.
3. İlgili kolonları seçer: `order_value` ve `num_items`.
4. `order_value` ve `num_items` arasındaki ilişkiyi görselleştirir.
5. Verilere Doğrusal Regresyon modeli uyarlar.
6. Modeli Ortalama Kare Hata (MSE) ile değerlendirir.
7. Yeni sipariş değerleri (ör. 10, 20, 50) için ürün sayısını tahmin eder.

### Sonuçlar:
- **Korelasyon:** `order_value` ve `num_items` arasındaki korelasyon hesaplanır ve yazdırılır.
- **MSE:** Ortalama Kare Hata, modelin ürün sayısını ne kadar iyi tahmin ettiğini gösterir.
- **Yeni Veri Tahminleri:** Belirtilen `order_value` girdileri için model şunları tahmin eder:
  - 10: 1 ürün
  - 20: 2 ürün
  - 50: 4 ürün

### Analiz:
Sonuçlar, `order_value` ve `num_items` arasında net bir doğrusal ilişki olduğunu göstermektedir. Düşük MSE, modelin bu ilişkiyi etkili bir şekilde yakaladığını göstermektedir.

---

## 2. **MusteriPuanTahmini.py**
### Amaç:
Bu dosya, teslimat süresine bağlı olarak bir müşteri puanının iyi (`iyi`) veya kötü (`kötü`) olduğunu Lojistik Regresyon kullanarak tahmin eder.

### İş Akışı:
1. `restaurant_sales_data.csv` dosyasından verileri okur ve temizler.
2. İlgili kolonları seçer: `delivery_time_minutes` ve `customer_rating`.
3. `customer_rating` değerlerini şu şekilde kategorize eder: 4-5 arası `iyi`, 1-3 arası `kötü`.
4. Verileri eğitim ve test setlerine ayırır.
5. Bir Lojistik Regresyon modeli eğitir.
6. Modeli doğruluk ve sınıflandırma raporu ile değerlendirir.
7. Yeni bir teslimat süresi girdisi (ör. 30 dakika) için müşteri puanını tahmin eder.

### Sonuçlar:
- **Doğruluk:** Model, %51 doğruluk elde eder.
- **Sınıflandırma Raporu:** Her iki sınıf için precision, recall ve F1-skoru içerir.
- **Yeni Veri Tahmini:** 30 dakikalık teslimat süresi için tahmin: `kötü`

### Analiz:
Bu model %51 doğruluk oranına sahiptir. Dolayısıyla çok güvenilir sonuçlar vereceğini söyleyemeyiz. Daha doğru sonuçlar için SMOTE ile yapay veri üretilerek azınlıkta olan verilerin miktarı arttırılarak model yeniden eğitilebilir. Bu sayede daha yüksek bir F1-score ve doğruluk elde edilebilir. Tabii SMOTE kullanırken verinin genel yapısını bozmamaya ve çok fazla yapay veri enjekte etmemeye dikkat edilmelidir. Çünkü bu bizi elde etmek istediğimiz sonuçlardan uzaklaştırabilir.
---

## 3. **FiyatTahmini.py**
### Amaç:
Bu dosya, çeşitli özelliklere dayalı olarak tatlı siparişlerinin fiyat kategorisini (`ucuz`, `orta`, `pahalı`) Destek Vektör Makineleri (SVM) kullanarak tahmin eder.

### İş Akışı:
1. `restaurant_sales_data.csv` dosyasından verileri okur ve temizler.
2. `items` sütununda "Dessert" içeren siparişleri filtreler.
3. `order_value` değerlerini şu şekilde kategorize eder: `ucuz`, `orta`, `pahalı`.
4. Özellikleri hazırlar: `num_items`, `delivery_time_minutes`, `hour`, `is_peak_hour`, `is_weekend`.
5. Hedef etiketlerini encode eder ve verileri eğitim ve test setlerine ayırır.
6. Özellikleri StandardScaler ile ölçeklendirir.
7. Doğrusal çekirdekli bir SVM modeli eğitir.
8. Modeli doğruluk ve sınıflandırma raporu ile değerlendirir.

### Sonuçlar:
- **Doğruluk:** Model, 87% doğruluk elde eder.
- **Sınıflandırma Raporu:** Her fiyat kategorisi için precision, recall ve F1-skoru içerir.

### Analiz:
SVM modeli, tatlı siparişlerini fiyat kategorilerine etkili bir şekilde sınıflandırır. Görüldüğü üzere yüksek bir doğruluk oranı var. Ama yüksek doğruluk oranı modelin her zaman düzgün öğrendiği anlamına gelmez. Doğruluk oranını başka metriklerle de desteklemek gerekir. Bu model için F1-score değerleri `ucuz` için %82, `orta` için %76, `pahalı` için %93 çıkmaktadır. Yani bu modelde hem doğruluk oranının yüksek olduğunu hem de bu oranın yanıltıcı olmadığını görmüş oluyoruz.

---

## Genel Gözlemler:
1. **Veri Kalitesi:**
   - Eksik değerler satır silme yöntemiyle ele alınmıştır, bu da veri kaybına yol açabilir. Gelecekteki yinelemelerde daha sağlam modeller için imputasyon teknikleri kullanılabilir.

2. **Özellik Seçimi:**
   - Her dosyada ilgili özellikler seçilmiş ve bu durum model performansını ve yorumlanabilirliğini artırmıştır.

3. **Model Performansı:**
   - Doğrusal Regresyon güvenilir performans göstermiştir.
   - Lojistik Regresyon çok da güvenilir olmayan bir performans göstermiştir. Dolayısıyla model tekrar gözden geçirilmelidir.
   - SVM, çok sınıflı bir hedef için güçlü bir sınıflandırma sağlamıştır. Güvenilir sonuçlar vermektedir.

4. **Yapılabilecek İyileştirmeler:**
   - Karşılaştırma için gelişmiş algoritmalar (ör. Random Forest, Gradient Boosting) kullanılabilir..
   - Daha iyi doğruluk için optimizasyon yapılabilir.
   - Daha sağlam performans metrikleri için çapraz doğrulama kullanılabilir.

---

## Gereksinimler:
- Python 3.x
- Kütüphaneler:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

Aşağıdaki komutla bağımlılıkları yükleyebilirsiniz:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Kurulum:
1. Depoyu klonlayın:
   ```bash
   git clone https://github.com/furkan-ylcn/TakeawayOrderDataAnalysis.git
   ```
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas scikit-learn matplotlib numpy
   ```

---

## Kullanım:
1. `restaurant_sales_data.csv` dosyasını betiklerle aynı dizine yerleştirin.
2. İlgili betiği çalıştırın:
   ```bash
   python <dosya_adı>.py
   ```
   `<dosya_adı>` yerine dosya adını yazın (ör. `UrunSayisiTahmini.py`).

3. Konsoldaki çıktıları ve görselleştirmeleri inceleyin.

---

## Sonuç:
Bu dosyalar, makine öğrenimi modellerinin tahmin analizi için pratik uygulamalarını göstermektedir. Elde edilen bilgiler, restoranların operasyonlarını optimize etmelerine, müşteri memnuniyetini artırmalarına ve veri odaklı kararlar almalarına yardımcı olabilir.

---

## Video Anlatımı:
Dosyaların içeriğini anlattığım videoya linken ulaşabilirsiniz: https://drive.google.com/file/d/1LGDv9mHGT6WrZ2z9BNxlDulhCNyyanWL/view?usp=sharing

---

