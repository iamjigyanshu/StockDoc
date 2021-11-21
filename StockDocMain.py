import streamlit as st
import datetime
import pandas_datareader as dt

import pandas as pd
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

#users will choose the start and end date
START = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
END = st.sidebar.date_input("End date", datetime.date(2021, 1, 31))

#setting the heading for web app
st.title('StockDoc')
ticker_list = pd.read_csv("/Users/SpyJigu/downloads/s&p500.csv")
#list of all the s&p500 stocks are given to users to choose from
selected_stock = st.sidebar.selectbox('Stock', ticker_list)


n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    #stock data is loaded
    data = dt.DataReader(ticker,'yahoo',START,END)
    data.reset_index(inplace=True)
    return data

    
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

s = yf.Ticker(selected_stock)


st.subheader('Showing for {0}'.format(selected_stock))
#setting different font
def subheader(url):
     st.markdown(f'<p style="color:#D3D3D3;font-size:16px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
subheader("Sector: "+s.info['sector'])
def subheader2(url):
     st.markdown(f'<p style="color:#F5F5F5;font-size:14px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
pe,mk,dy = s.info['trailingPE'],s.info['marketCap'],s.info['dividendYield']

#some stock infos
st.sidebar.write("PE: {0} | Market Cap: {1}M |".format(round(pe,2),round(mk/1000000,2)))
st.sidebar.write("Dividen Yield: {0}%".format(round(dy*100,2)))
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)

st.plotly_chart(fig1,template="plotly")



st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


