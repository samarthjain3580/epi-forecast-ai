import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

st.set_page_config(page_title="Epidemic Forecast AI", layout="centered")

# HEADER
st.markdown("""
# 🦠 Epidemic Forecast AI
### 📊 AI-powered epidemic prediction system
""")

st.markdown("---")

# INPUTS
country = st.selectbox("🌍 Select Country", ["india", "usa", "brazil", "uk"])
disease = st.selectbox("🦠 Select Disease", ["covid", "flu", "dengue"])

st.markdown("---")

with st.spinner("Fetching live predictions..."):
    try:

        api_base = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000")
        url = f"{api_base}/predict?country={country}&disease={disease}"
        response = requests.get(url)
        data = response.json()

        preds = data["predicted_cases_next_7_days"]
        past = data["past_30_days"]

        st.success(f"Prediction for {country.upper()} ({disease})")

        # 📅 DATES
        today = datetime.today()

        future_dates = [
            (today + timedelta(days=i+1)).strftime("%b %d")
            for i in range(len(preds))
        ]

        # 📅 CARDS
        st.subheader("📅 7-Day Forecast")

        for i, value in enumerate(preds):
            change = round(value - preds[i-1], 2) if i > 0 else 0
            trend = "📈 Increasing" if i > 0 and value > preds[i-1] else "📊 Stable"

            color = "#16a34a" if change >= 0 else "#dc2626"

            st.markdown(f"""
            <div style="
                padding:16px;
                border-radius:12px;
                background-color:#111827;
                margin-bottom:12px;
                border:1px solid #1f2937;
            ">
                <h4>📅 Day {i+1} ({future_dates[i]})</h4>
                <h2 style="color:{color};">{value:.2f}</h2>
                <p>🔍 Trend: {trend}</p>
                <p>📊 Change: {change:+.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        # 📈 GRAPH (PAST + FUTURE)
        st.subheader("📈 Trend Forecast (Past + Future)")

        past_dates = [
            (today - timedelta(days=len(past)-i)).strftime("%b %d")
            for i in range(len(past))
        ]

        future_dates = [
            (today + timedelta(days=i+1)).strftime("%b %d")
            for i in range(len(preds))
        ]

        past_df = pd.DataFrame({
            "Date": past_dates,
            "Past": past
        })

        future_df = pd.DataFrame({
            "Date": future_dates,
            "Forecast": preds
        })

        combined = pd.merge(past_df, future_df, on="Date", how="outer")

        st.line_chart(combined.set_index("Date"))

        # 🔮 INSIGHT
        st.subheader("🔮 Key Insight")

        final_val = preds[-1]
        st.success(f"📌 Expected cases after 7 days: {final_val:.2f}")

        if preds[-1] > preds[0]:
            st.warning("⚠️ Upward trend detected — cases may increase")
        else:
            st.success("✅ Stable or decreasing trend")

        # 🗺️ MAP
        st.subheader("🗺️ Location Overview")

        coords = {
            "india": [28.6139, 77.2090],
            "usa": [37.0902, -95.7129],
            "brazil": [-14.2350, -51.9253],
            "uk": [51.5074, -0.1278]
        }

        lat, lon = coords.get(country, [28.6, 77.2])
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}))

    except Exception as e:
        st.error("⚠️ Backend connection failed")
        st.text(str(e))

st.markdown("---")

# ABOUT
st.markdown("""
### ℹ️ About

- LSTM Deep Learning  
- Time-Series Forecasting  
- Flask API  
- Streamlit Dashboard  

Live epidemic prediction system.
""")

# AUTO REFRESH
time.sleep(60)
st.rerun()