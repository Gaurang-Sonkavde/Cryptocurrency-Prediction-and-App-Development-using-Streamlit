import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date
import pystan as ps
from PIL import Image
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import yfinance as yf
import pickle

st.set_page_config(layout='wide')
image = Image.open('Coin.png')
st.image(image, width = 500)



START = "2017-11-9"
TODAY = "2021-7-28"

st.title("Cryptocurrency Prediction App")

st.markdown("""
I have made this mainly for Binance-Coin (BND-USD) but we also have Bitcoin(BTC-USD) and Etheriuum (ETH-USD)
""")

Coin = ("BNB-USD","BTC-USD","ETH-USD")




selected_Crypto = st.sidebar.selectbox("Select Crypto Currency for Prediction",Coin)

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

model1 = Prophet()
model1.fit(df_train)
future = model1.make_future_dataframe(periods=period)
forecast = model1.predict(future)

st.subheader('Forecast Crypto Price upto 1 Year')
st.write(forecast.tail())

#Plot Forecasted Data

st.write("Forecasted Data for an Year")
fig1 = plot_plotly(model1, forecast)
st.plotly_chart(fig1)


st.write("forecast components")
fig2 = model1.plot_components(forecast)
st.write(fig2)

pickle.dump(forecast,open('Crypto_classifier.pkl','wb'))