import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from datetime import date
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf

os.chdir(r'C:\Users\gaura\Downloads')

Binance= pd.read_csv('Binance Coin - Historic data.csv')

print(Binance)

Binance.describe()

START = "2017-11-9"
TODAY = "2021-7-28"


st.title("My Crypto Predictor")

Coin = ("BNB-USD","BTC-USD","ETH-USD")

selected_Crypto = st.selectbox("Select Crypto Currency for Prediction",Coin)

n_years = st.slider("Years of Prediction",1 ,3)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Load data...")
data = load_data(selected_Crypto)
data_load_state.text("Loading Data...done!")

st.subheader('Raw Data')
st.write(data.head())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='Crypto_Opening'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Crypto_Closing'))
    fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

#Forecasting

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date":"ds" , "Close":"y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader('Forecast Crypto Price upto 1 Year')
st.write(forecast.tail())

#Plot Forecasted Data

st.write("Forecasted Data for an Year")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write("forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)


