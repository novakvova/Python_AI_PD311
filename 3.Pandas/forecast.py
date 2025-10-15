import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

# ---------- 1. Перевірка наявності файлу ----------
CSV_FILE = "usd_uah.csv"

if not os.path.exists(CSV_FILE):
    print("⚙️  Файл usd_uah.csv не знайдено — завантажую курс з НБУ...")
    end_date = datetime.now().date()
    #проміжок між поточного датою і назад 365 днів
    start_date = end_date - timedelta(days=365)

    url = f"https://bank.gov.ua/NBU_Exchange/exchange_site?start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&valcode=usd&sort=exchangedate&json"
    data = requests.get(url).json()

    df = pd.DataFrame(data)[["exchangedate", "rate"]]
    df.columns = ["date", "usd_uah"]
    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y")
    df["usd_uah"] = pd.to_numeric(df["usd_uah"], errors="coerce")

    df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    print("✅ Файл usd_uah.csv створено успішно!")

# ---------- 2. Читання CSV ----------
df = pd.read_csv(CSV_FILE)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "usd_uah"])
df = df.sort_values("date").reset_index(drop=True)
df["usd_uah"] = pd.to_numeric(df["usd_uah"], errors="coerce")



print(f"📊 Завантажено {len(df)} рядків з {CSV_FILE}")

# ---------- 3. Підготовка даних ----------
n_lags = 14
forecast_days = 10
test_size = 0.15

ts = df[["date", "usd_uah"]].copy()
ts.columns = ["date", "value"]

# Створення лагів
for lag in range(1, n_lags + 1):
    ts[f"lag_{lag}"] = ts["value"].shift(lag)

ts = ts.dropna().reset_index(drop=True)

n = len(ts)
test_n = max(1, int(n * test_size))
train = ts.iloc[:-test_n]
test = ts.iloc[-test_n:]

X_train = train[[f"lag_{i}" for i in range(1, n_lags+1)]].values
y_train = train["value"].values
X_test = test[[f"lag_{i}" for i in range(1, n_lags+1)]].values
y_test = test["value"].values

# ---------- 4. Навчання ----------
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# ---------- 5. Перевірка якості ----------
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"📈 RMSE на тесті: {rmse:.4f}")

# ---------- 6. Прогноз на 10 днів ----------
last_row = ts.tail(1)[[f"lag_{i}" for i in range(1, n_lags+1)]].values.flatten()
forecast = []
current_features = last_row.copy()

for _ in range(forecast_days):
    pred = model.predict(current_features.reshape(1, -1))[0]
    forecast.append(pred)
    current_features = np.roll(current_features, 1)
    current_features[0] = pred

last_date = ts["date"].iloc[-1]
future_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

df_forecast = pd.DataFrame({"date": future_dates, "forecast": forecast})

# ---------- 7. Графік ----------
plt.figure(figsize=(12,6))
plt.plot(ts["date"], ts["value"], label="Фактичний курс", color="blue")
plt.plot(df_forecast["date"], df_forecast["forecast"], marker="o", linestyle="--", color="orange", label="Прогноз")
plt.axvline(ts["date"].iloc[-1], color="gray", linestyle=":", label="Кінець історії")
plt.xlabel("Дата")
plt.ylabel("Курс USD → UAH")
plt.title("Прогноз курсу USD → UAH на 10 днів вперед")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- 8. Вивід прогнозу ----------
print("\n🔮 Прогноз на 10 днів:")
print(df_forecast.to_string(index=False, formatters={"forecast": "{:.4f}".format}))
