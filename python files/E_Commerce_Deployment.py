import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load(r'C:\Users\moham\Downloads\DEPI\2_Project\DEPI_PROJECT\model.pkl')
scaler = joblib.load(r'C:\Users\moham\Downloads\DEPI\2_Project\DEPI_PROJECT\scaler.pkl')
encoder = joblib.load(r'C:\Users\moham\Downloads\DEPI\2_Project\DEPI_PROJECT\encoder.pkl')
category_means = joblib.load(r'C:\Users\moham\Downloads\DEPI\2_Project\DEPI_PROJECT\category_means.pkl')
train_columns = joblib.load(r"C:\Users\moham\Downloads\DEPI\2_Project\DEPI_PROJECT\train_columns.pkl")

SCALE_COLS = ["Tenure", "WarehouseToHome", "NumberOfAddress", 
             "CashbackAmount", "AvgOrdersPerHour", "AvgCashBackperCategory",
             "DaySinceLastOrder", "OrderAmountHikeFromlastYear"]

ONE_HOT = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender", 
           "PreferedOrderCat", "MaritalStatus"]

st.set_page_config(page_title="E-Commerce Churn Prediction", layout="centered")
st.title("🛍️ E-Commerce Churn Prediction App")
st.subheader("Enter Customer Details")

CustomerID = st.text_input("Customer ID")
Tenure = st.slider("Tenure (months)", 0, 100, 1)
PreferredLoginDevice = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer", "Phone"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
WarehouseToHome = st.slider("Distance from Warehouse to Home (km)", 0, 100, 10)
PreferredPaymentMode = st.selectbox("Preferred Payment Mode", ["Credit Card", "Debit Card", "UPI", "COD", "CC", "Cash on Delivery", "E wallet"])
Gender = st.selectbox("Gender", ["Male", "Female"])
HourSpendOnApp = st.slider("Hours Spent on App per Day", 0, 20, 3)
NumberOfDeviceRegistered = st.slider("Number of Devices Registered", 1, 10, 4)
PreferedOrderCat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Mobile", "Grocery", "Others"])
SatisfactionScore = st.slider("Satisfaction Score (1 to 5)", 1, 5, 3)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfAddress = st.slider("Number of Addresses", 1, 20, 2)
Complain = 1 if st.checkbox("Complain Raised Last Month?") else 0
OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike From Last Year (%)", min_value=0.0, value=10.0)
CouponUsed = st.slider("Number of Coupons Used", 0, 10, 2)
OrderCount = st.slider("Order Count (Last Month)", 1, 10, 2)
DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0, value=3)
CashbackAmount = st.number_input("Cashback Amount (Last Month)", min_value=0.0, value=0.0)

AvgOrdersPerHour = HourSpendOnApp / OrderCount
AvgCashBackperCategory = category_means.get(PreferedOrderCat, 0)
LoyaltyScore = 1 if Tenure <= 12 else (2 if Tenure <= 24 else 3)
OrderFrequencyBin = 1 if OrderCount <= 2 else (2 if OrderCount <= 4 else 3)

input_data = pd.DataFrame([{
    'Tenure': Tenure,
    'PreferredLoginDevice': PreferredLoginDevice,
    'CityTier': CityTier,
    'WarehouseToHome': WarehouseToHome,
    'PreferredPaymentMode': PreferredPaymentMode,
    'Gender': Gender,
    'HourSpendOnApp': HourSpendOnApp,
    'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
    'PreferedOrderCat': PreferedOrderCat,
    'SatisfactionScore': SatisfactionScore,
    'MaritalStatus': MaritalStatus,
    'NumberOfAddress': NumberOfAddress,
    'Complain': Complain,
    'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
    'CouponUsed': CouponUsed,
    'OrderCount': OrderCount,
    'DaySinceLastOrder': DaySinceLastOrder,
    'CashbackAmount': CashbackAmount,
    'AvgOrdersPerHour': AvgOrdersPerHour,
    'AvgCashBackperCategory': AvgCashBackperCategory,
    'LoyaltyScore': LoyaltyScore,
    'OrderFrequencyBin': OrderFrequencyBin
}])


for col in SCALE_COLS:
    input_data[col] = np.log1p(input_data[col])

input_data[SCALE_COLS] = scaler.transform(input_data[SCALE_COLS])

encoded_data = encoder.transform(input_data)

for col in train_columns:
    if col not in encoded_data.columns:
        encoded_data[col] = 0

final_data = encoded_data[train_columns]

if st.button("Predict Churn"):
    try:
        prediction = model.predict(final_data)
        st.success(f"Prediction: {'Churn 🔴' if prediction[0] == 1 else 'Not Churn 🟢'}")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")