import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# CSV dosyasından verileri oku
file_path = 'restaurant_sales_data.csv'
data = pd.read_csv(file_path)

# Datayı temizle

# Boş satır var mı diye kontrol et
data.isnull().sum()

# Boş satırları temizle
data.dropna()

# İşimize yarayan kolonları tanımla
istenilen_kolonlar = ['delivery_time_minutes', 'customer_rating']

# Diğer kolonları at
data = data[istenilen_kolonlar]

# customer_rating 1-3 arası ise kötü, 4-5 arası ise iyi olacak şekilde etiketle
data.loc[:, 'customer_rating'] = data['customer_rating'].apply(lambda x: "kötü" if x <= 3 else "iyi")

# Lojistik regresyon modeli oluşturma
X = data['delivery_time_minutes'].values.reshape(-1, 1)
y = data['customer_rating'].values

# Veriyi eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Yeni veri ile tahmin yapma
new_data = np.array([30]).reshape(-1, 1)
y_pred_new = model.predict(new_data)

print("Yeni veri ile tahmin:", y_pred_new)