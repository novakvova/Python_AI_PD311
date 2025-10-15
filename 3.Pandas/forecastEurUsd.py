import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests

CSV_FILE = "eur_usd.csv"

# ---------- 1. Завантаження даних (якщо немає CSV) ----------
if not os.path.exists(CSV_FILE):
    print("⚙️ Завантажую курс EUR→USD за останній рік...")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    url = f"https://api.exchangerate.host/timeseries?start_date={start_date}&end_date={end_date}&base=EUR&symbols=USD"
    response = requests.get(url)
    data = response.json()
    if not data.get("rates"):
        raise ValueError("Не вдалося отримати дані з exchangerate.host")
    
    rows = []
    for date_str, value in data["rates"].items():
        rate = value.get("USD") if isinstance(value, dict) else value
        rows.append({
            "Date": date_str,
            "Open": rate,
            "High": rate,
            "Low": rate,
            "Close": rate,
            "Change(Pips)": 0.0,
            "Change(%)": 0.0
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna().sort_values("Date").reset_index(drop=True)
    
    # Заповнюємо пропущені дати інтерполяцією Close
    all_dates = pd.date_range(df["Date"].min(), df["Date"].max())
    df = df.set_index("Date").reindex(all_dates)
    df["Close"] = df["Close"].interpolate(method="linear")
    df[["Open","High","Low","Change(Pips)","Change(%)"]] = df[["Close"]]*0
    df = df.rename_axis("Date").reset_index()
    
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")
    print(f"✅ Файл {CSV_FILE} створено успішно! {len(df)} рядків")

# ---------- 2. Читання CSV ----------
df = pd.read_csv(CSV_FILE, usecols=range(7))  # беремо тільки перші 7 колонок
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M", errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
print(f"📊 Завантажено {len(df)} рядків з {CSV_FILE}")

# ---------- 3. Підготовка даних ----------
n_lags = 14
forecast_days = 10
test_size = 0.15

ts = df[["Date", "Close"]].copy()
ts.columns = ["date", "value"]

# Створюємо лаги
for lag in range(1, n_lags + 1):
    ts[f"lag_{lag}"] = ts["value"].shift(lag)

ts = ts.dropna().reset_index(drop=True)

# Розділяємо на train/test
n = len(ts)
test_n = max(1, int(n * test_size))
train = ts.iloc[:-test_n]
test = ts.iloc[-test_n:]

X_train = train[[f"lag_{i}" for i in range(1, n_lags + 1)]].values
y_train = train["value"].values
X_test = test[[f"lag_{i}" for i in range(1, n_lags + 1)]].values
y_test = test["value"].values

# ---------- 4. Навчання ----------
model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# ---------- 5. Перевірка якості ----------
y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f"📈 RMSE на тесті: {rmse:.6f}")

# ---------- 6. Прогноз на 10 днів ----------
last_row = ts.tail(1)[[f"lag_{i}" for i in range(1, n_lags + 1)]].values.flatten()
forecast = []
current_features = last_row.copy()

for _ in range(forecast_days):
    pred = model.predict(current_features.reshape(1, -1))[0]
    forecast.append(pred)
    current_features = np.roll(current_features, 1)
    current_features[0] = pred

last_date = ts["date"].iloc[-1]
future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
df_forecast = pd.DataFrame({"date": future_dates, "forecast": forecast})

# ---------- 7. Графік ----------
plt.figure(figsize=(12, 6))
plt.plot(ts["date"], ts["value"], label="Фактичний курс Close", color="blue")
plt.plot(df_forecast["date"], df_forecast["forecast"], marker="o", linestyle="--", color="orange", label="Прогноз")
plt.axvline(ts["date"].iloc[-1], color="gray", linestyle=":", label="Кінець історії")
plt.xlabel("Дата")
plt.ylabel("Курс EUR → USD (Close)")
plt.title("Прогноз курсу EUR → USD на 10 днів вперед (Close)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- 8. Вивід прогнозу ----------
print("\n🔮 Прогноз на 10 днів (Close):")
print(df_forecast.to_string(index=False, formatters={"forecast": "{:.5f}".format}))
