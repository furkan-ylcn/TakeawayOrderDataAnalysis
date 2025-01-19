import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# CSV dosyasından verileri oku
file_path = 'restaurant_sales_data.csv'
data = pd.read_csv(file_path)

# Datayı temizle

# Boş satır var mı diye kontrol et
data.isnull().sum()

# Boş satırları temizle
data.dropna()

# İşimize yarayan kolonları tanımla
istenilen_kolonlar = ['order_value', 'num_items']

# Diğer kolonları at
data = data[istenilen_kolonlar]

# Data grafiğini çiz
plt.scatter(data['order_value'], data['num_items'])

# Korelasyon katsayısını hesapla
print(data['order_value'].corr(data['num_items']))

# Linear regression modelini olustur
X = data['order_value'].values.reshape(-1, 1)
y = data['num_items'].values

# Veriyi eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Modelin grafiğini çiz
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Order Value')
plt.ylabel('Number of Items')

# Mean Squared Error hesapla
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Yeni data ile modeli test et
new_data = pd.DataFrame({'order_value': [10, 20, 50]})
new_data_pred = model.predict(new_data['order_value'].values.reshape(-1, 1))
print(new_data_pred)
