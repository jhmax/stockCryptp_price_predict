import streamlit as st
import datetime

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objects as go

# START = "2015-01-01"
st.title("Predict Price of Stock/Crypto")
st.write("#")

st.sidebar.subheader("Choose Ticker")
ticker = st.sidebar.text_input("Ticker", "BTC-USD")
start_date = st.sidebar.date_input("Start Date", datetime.date(2015,1,1))
end_date = st.sidebar.date_input("End Date")


@st.cache
def load_data():
    df = yf.download(ticker, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    return df

## DISPLAY RAW DATA
st.subheader("Raw data")
data_load_state = st.text("Load data...")
data = load_data()
data_load_state.text("Loading data...done!")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting
st.subheader("Forecast data")

n_years = st.slider("Years of prediction", 1,10)
period = n_years * 365

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.write(forecast.tail())


fig1 = plot_plotly(m,forecast)
st.plotly_chart(fig1)



st.write("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
