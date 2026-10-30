# Car Price Prediction System

Streamlit app that predicts used-car resale prices (regression).

## Run locally
```
pip install -r requirements.txt
python make_sample_data.py   # optional: only if data/car_data.csv is missing
python train_model.py        # trains and saves model/
streamlit run app.py
```

## Use the real dataset
Download CarDekho "car data.csv" from Kaggle, rename to `car_data.csv`, put it in `data/`,
then run `python train_model.py`. Required columns: Year, Selling_Price, Present_Price,
Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to share.streamlit.io -> New app -> select repo, branch, main file `app.py`.
3. Deploy. (If the saved model is incompatible, the app retrains itself automatically.)
