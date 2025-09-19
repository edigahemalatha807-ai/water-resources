# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import smtplib
from email.mime.text import MIMEText
import requests

st.set_page_config(layout="wide", page_title="DWLR Groundwater Dashboard")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("sample_dwlr.csv", parse_dates=["datetime"])
    return df

df = load_data()

st.title("💧 Real-Time Groundwater Monitoring (DWLR)")

# ---------------- Station Selector ----------------
stations = df["station_id"].unique()
selected_station = st.selectbox("Select Station", stations)

filtered_df = df[df["station_id"] == selected_station]

# ---------------- Email Alert Function ----------------
def send_email_alert(message):
    sender = "your_email@gmail.com"
    password = "your_app_password"  # Gmail App Password (not normal password)
    receiver = "receiver_email@gmail.com"

    msg = MIMEText(message)
    msg["Subject"] = "🚨 DWLR Water Level Alert"
    msg["From"] = sender
    msg["To"] = receiver

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print("✅ Email sent successfully")
    except Exception as e:
        print("❌ Email failed:", e)

# ---------------- Fast2SMS Alert Function ----------------
def send_sms_alert(message, phone_number="+91XXXXXXXXXX"):
    url = "https://www.fast2sms.com/dev/bulkV2"
    payload = {
        "sender_id": "TXTIND",
        "message": message,
        "route": "v3",
        "numbers": phone_number
    }
    headers = {
        "authorization": "YOUR_FAST2SMS_API_KEY",  # Replace with Fast2SMS API Key
        "Content-Type": "application/x-www-form-urlencoded"
    }

    try:
        response = requests.post(url, data=payload, headers=headers)
        if response.status_code == 200:
            print("✅ SMS sent successfully")
        else:
            print("❌ SMS failed:", response.text)
    except Exception as e:
        print("⚠️ Error sending SMS:", e)

# ---------------- Alerts System ----------------
st.subheader("🚨 Water Level Alerts")

latest_level = filtered_df.sort_values("datetime").iloc[-1]["water_level"]

LOW_THRESHOLD = 2.0   # meters
HIGH_THRESHOLD = 10.0 # meters

if latest_level < LOW_THRESHOLD:
    alert_message = f"⚠️ ALERT: Water level critically LOW ({latest_level:.2f} m)"
    st.error(alert_message)
    send_email_alert(alert_message)
    send_sms_alert(alert_message, "+91XXXXXXXXXX")

elif latest_level > HIGH_THRESHOLD:
    alert_message = f"⚠️ ALERT: Water level unusually HIGH ({latest_level:.2f} m)"
    st.warning(alert_message)
    send_email_alert(alert_message)
    send_sms_alert(alert_message, "+91XXXXXXXXXX")

else:
    st.success(f"✅ Water level is Normal ({latest_level:.2f} m)")

# ---------------- Line Chart ----------------
fig = px.line(
    filtered_df,
    x="datetime",
    y="water_level",
    title=f"Water Level Trend - {selected_station}",
    labels={"datetime": "Date & Time", "water_level": "Water Level (m)"},
    markers=True
)
st.plotly_chart(fig, use_container_width=True)

# ---------------- Map of All Stations ----------------
st.subheader("📍 Station Locations")
map_fig = px.scatter_mapbox(
    df,
    lat="lat",
    lon="lon",
    color="station_id",
    size="water_level",
    hover_name="station_id",
    hover_data={"lat": False, "lon": False, "water_level": True},
    zoom=4,
    height=500
)
map_fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(map_fig, use_container_width=True)

# ---------------- Bar Chart (Average Water Levels) ----------------
st.subheader("📊 Average Water Level per Station")
avg_levels = df.groupby("station_id")["water_level"].mean().reset_index()
bar_fig = px.bar(
    avg_levels,
    x="station_id",
    y="water_level",
    title="Average Water Levels",
    labels={"water_level": "Avg Water Level (m)", "station_id": "Station"},
    text_auto=True
)
st.plotly_chart(bar_fig, use_container_width=True)

# ---------------- Trend Analysis (Min & Max) ----------------
st.subheader("📉 Water Level Statistics")
stats = df.groupby("station_id")["water_level"].agg(["min", "max", "mean"]).reset_index()
st.dataframe(stats, use_container_width=True)

# ---------------- Forecasting with Prophet ----------------
st.subheader("🔮 Forecast Future Water Levels")

prophet_df = filtered_df[["datetime", "water_level"]].rename(
    columns={"datetime": "ds", "water_level": "y"}
)

m = Prophet()
m.fit(prophet_df)

future = m.make_future_dataframe(periods=30, freq="D")
forecast = m.predict(future)

forecast_fig = plot_plotly(m, forecast)
forecast_fig.update_layout(title=f"Forecasted Water Levels - {selected_station}")
st.plotly_chart(forecast_fig, use_container_width=True)

# ---------------- Download Option ----------------
st.subheader("📤 Download Filtered Data")
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV for Selected Station",
    data=csv,
    file_name=f"{selected_station}_water_data.csv",
    mime="text/csv",
)
