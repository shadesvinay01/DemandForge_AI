
import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go

st.set_page_config(page_title="DemandForge AI", layout="wide")
st.title("📈 DemandForge AI")
st.markdown("**Advanced Time Series Forecasting Platform with Uncertainty**")

st.sidebar.header("Configuration")
forecast_days = st.sidebar.slider("Forecast Horizon (Days)", 30, 730, 365)
confidence = st.sidebar.selectbox("Confidence Interval", [0.80, 0.90, 0.95], index=2)

# Load Data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv"
    df = pd.read_csv(url)
    df.columns = ['ds', 'y']
    return df

df = load_data()

tab1, tab2 = st.tabs(["Forecasting", "Anomaly Detection"])

with tab1:
    if st.button("🚀 Train & Forecast", type="primary"):
        with st.spinner("Training Prophet Model..."):
            model = Prophet(interval_width=confidence, yearly_seasonality=True, weekly_seasonality=True)
            model.fit(df)
            
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Actual Data", line=dict(color="#1f77b4")))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="Forecast", line=dict(color="#ff7f0e")))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name="Lower Bound", line=dict(dash='dash', color="#d62728")))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name="Upper Bound", line=dict(dash='dash', color="#2ca02c")))
            
            fig.update_layout(title=f"Time Series Forecast with {int(confidence*100)}% Uncertainty", height=650)
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"✅ Forecast generated for next {forecast_days} days!")
            
            # Download
            csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button("📥 Download Full Forecast", csv, "demand_forecast.csv", "text/csv")

with tab2:
    st.subheader("Anomaly Detection")
    if st.button("🔍 Detect Anomalies"):
        model = Prophet().fit(df)
        forecast = model.predict(df)
        df['yhat'] = forecast['yhat']
        df['residual'] = df['y'] - df['yhat']
        
        threshold = 2.5 * df['residual'].std()
        anomalies = df[abs(df['residual']) > threshold]
        
        st.write(f"**Found {len(anomalies)} anomalies** in the data")
        st.dataframe(anomalies[['ds', 'y', 'yhat', 'residual']])
