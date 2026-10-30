"""Generates a synthetic dataset in the CarDekho (Kaggle) schema so the app runs out of the box.
Replace data/car_data.csv with the real Kaggle CSV for real results."""
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 1500
names = ["swift", "city", "i20", "creta", "baleno", "innova", "fortuner", "alto", "verna", "ertiga", "xuv500", "bullet", "activa", "pulsar"]
year = rng.integers(2005, 2025, n)
present = np.round(np.exp(rng.normal(1.9, 0.8, n)).clip(0.5, 40), 2)
kms = (np.clip((2026 - year) * rng.normal(11000, 3500, n), 500, 300000)).astype(int)
fuel = rng.choice(["Petrol", "Diesel", "CNG"], n, p=[0.6, 0.36, 0.04])
seller = rng.choice(["Dealer", "Individual"], n, p=[0.65, 0.35])
trans = rng.choice(["Manual", "Automatic"], n, p=[0.85, 0.15])
owner = rng.choice([0, 1, 3], n, p=[0.9, 0.09, 0.01])

age = 2026 - year
price = present * np.exp(-0.11 * age) * (1 - kms / 900000)
price *= np.where(fuel == "Diesel", 1.08, np.where(fuel == "CNG", 0.95, 1.0))
price *= np.where(trans == "Automatic", 1.12, 1.0)
price *= np.where(seller == "Individual", 0.9, 1.0)
price *= np.where(owner > 0, 0.9, 1.0)
price *= rng.normal(1, 0.06, n)

df = pd.DataFrame({
    "Car_Name": rng.choice(names, n), "Year": year,
    "Selling_Price": np.round(price.clip(0.1), 2), "Present_Price": present,
    "Kms_Driven": kms, "Fuel_Type": fuel, "Seller_Type": seller,
    "Transmission": trans, "Owner": owner,
})
df.to_csv("data/car_data.csv", index=False)
print(df.shape)
