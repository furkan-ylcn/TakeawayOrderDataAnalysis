import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# CSV dosyasından verileri oku
file_path = 'restaurant_sales_data.csv'
data = pd.read_csv(file_path)

# Datayı temizle

# Boş satır var mı diye kontrol et
data.isnull().sum()

# Boş satırları temizle
data.dropna()

# DataFrame'i oluşturma
df = pd.DataFrame(data)

# "Dessert" içeren siparişleri seç
dessert_orders = df[df["items"].str.contains("Dessert", case=False, na=False)]

# "order_value"ya göre kategoriler oluştur
def categorize_order_value(value):
    if value <= 20:
        return "ucuz"
    elif 21 <= value <= 40:
        return "orta"
    else:
        return "pahalı"

# Hedef değişkeni oluştur
dessert_orders['order_category'] = dessert_orders["order_value"].apply(categorize_order_value)

# Özellikleri seç
X = dessert_orders[["num_items", "delivery_time_minutes", "hour", "is_peak_hour", "is_weekend"]]

# Hedef değişkeni (order_category) etiketle (ucuz, orta, pahalı)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(dessert_orders['order_category'])

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Verileri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM modelini oluştur ve eğit
svm_model = SVC(kernel="linear", random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Test verisi ile tahmin yap
y_pred = svm_model.predict(X_test_scaled)

# Accuracy hesapla
accuracy = svm_model.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)

# Sonuçları değerlendir
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))