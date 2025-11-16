import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("Taxi Demand Prediction App")

st.write("Upload the taxi_demand.xlsx file to train the model and make predictions.")

uploaded_file = st.file_uploader("Upload taxi_demand.xlsx", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.write("### Preview of Data")
    st.dataframe(df.head())

    # Convert datetime into ML features
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    X = df[["hour", "day", "month", "weekday"]]
    y = df["demand"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train Model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # Prediction Inputs
    st.write("### Make a Prediction")

    hour = st.slider("Hour of Day", 0, 23)
    day = st.slider("Day", 1, 31)
    month = st.slider("Month", 1, 12)
    weekday = st.slider("Weekday (0=Mon, 6=Sun)", 0, 6)

    input_data = pd.DataFrame([[hour, day, month, weekday]],
                              columns=["hour", "day", "month", "weekday"])

    prediction = model.predict(input_data)[0]

    st.write(f"### Predicted Taxi Demand: *{prediction:.2f}*")
