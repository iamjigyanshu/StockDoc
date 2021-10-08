import streamlit as st
import datetime
import yfinance as yf
from plotly import graph_objs as go
from urllib import request
import pandas as pd
import base64
from traverse import Traverse
from Stock_Sentiment_Analysis import Sentiment_Analyzer
from prediction import Predictor

st.set_page_config(
    page_title="Stock Sensei",
    page_icon="📈",
    layout="wide",
)

traverser = Traverse()
sentiment_analyzer = Sentiment_Analyzer()
predictor = Predictor(traverser)

@st.cache(allow_output_mutation=True)
def load_stock_data(ticker):
    data = yf.download(ticker,START,TODAY)
    stock_data = yf.Ticker(ticker)
    data.reset_index(inplace=True)
    return (data, stock_data)

@st.cache
def load_stock_list():
    url = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = request.urlopen(url)
    html = response.read()
    data = pd.read_html(html,header = None)
    df = data[0]
    return df

@st.cache
def process_data(data):
    data['SMA'] = data['Close'].rolling(window=30).mean()
    data['EMA'] = data['Close'].ewm(span = 20, adjust = False).mean()
    data['ShortEMA'] = data['Close'].ewm(span = 12, adjust = False).mean()
    data['LongEMA'] = data['Close'].ewm(span = 26, adjust = False).mean()
    data['MACD'] = data['ShortEMA'] - data['LongEMA']
    data['Signal'] = data['MACD'].ewm(span = 9, adjust = False).mean()
    #RSI
    delta = data['Close'].diff()
    delta = delta[1:]
    up = delta.copy()
    down = delta.copy()
    up[up<0] = 0
    down[down>0] = 0
    data['UP'] = up
    data['DOWN'] = down
    data['Avg_Gain'] = data['UP'].rolling(window=14).mean()
    data['Avg_Loss'] = abs(data['DOWN'].rolling(window=14).mean())
    RS = data['Avg_Gain'] / data['Avg_Loss']
    data['RSI'] = 100.0 - (100.0/(1.0 + RS))
    return data

st.title("Stock Sensei")
st.markdown("""
This is a Stock Analysis and Prediction Web Application that retrieves the list of the **S&P 500** (from Wikepedia) and its corresponding **stock opening and closing price** (year-to-date).
* **Python Libraries :** base64, pandas, numpy, datetime, streamlit, yfinance, plotly, tensorflow
* **Data Source :** [Wikepedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)      [Yahoo Finance](https://in.finance.yahoo.com/)         [finviz](https://finviz.com)
""")
st.write('---')
st.sidebar.header("User Input Features")

df = load_stock_list()
sector = df.groupby('GICS Sector')
sorted_sector_unique = sorted(df['GICS Sector'].unique())
selected_sectors = st.sidebar.multiselect('Sector',sorted_sector_unique)
df_selected_sectors = df[df['GICS Sector'].isin(selected_sectors)]
st.sidebar.write(f"Companies : {df_selected_sectors.shape[0] or df.shape[0]}")

START =st.sidebar.date_input(label="Enter Start Date",value=datetime.date(2017,1,1))
TODAY = datetime.date.today().strftime("%Y-%m-%d")

if len(selected_sectors) != 0:
    stocks = df_selected_sectors.Symbol.values 
else:
    stocks = df.Symbol.values
selected_stock = st.sidebar.selectbox("Select Stock",stocks)

data = pd.DataFrame()
processed_data = pd.DataFrame()

def filedownload(data,message):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        if message == "Raw":
            href = f'<a href="data:file/csv;base64,{b64}" download="{selected_stock}_RAW.csv">Download {message} CSV File</a>'
        elif message == "Processed":
            href = f'<a href="data:file/csv;base64,{b64}" download="{selected_stock}_PROCESSED.csv">Download {message} CSV File</a>'
        return href

if selected_stock != None:
    data, stock_details = load_stock_data(selected_stock)
    processed_data = process_data(data.copy())
    string_logo = f"<img src={stock_details.info['logo_url']}>"
    st.markdown(string_logo,unsafe_allow_html=True)
    st.header(stock_details.info["longName"])
    with st.beta_expander(label="Comapny Description"):
        st.info(stock_details.info["longBusinessSummary"])
    container = st.beta_container()
    col1, col2 = container.beta_columns([5,5])

    col1.subheader(f"Raw Data")
    col1.write(data.tail(50))
    col1.markdown(filedownload(data,"Raw"), unsafe_allow_html=True)

    col2.subheader(f"Processed Data")
    col2.write(processed_data.tail(50))
    col2.markdown(filedownload(processed_data,"Processed"), unsafe_allow_html=True)

    previous_averaged_volume = data['Volume'].iloc[1:4:1].mean()
    todays_volume = data['Volume'].iloc[-1]
    st.markdown(f"""
    **Today's Volume**           : {todays_volume}
    **Previous Averaged Volume** : {round(previous_averaged_volume,2)}
    """)
if data.empty == False:
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.layout.update(title_text="Time Series Chart", xaxis_rangeslider_visible=True,yaxis_title="USD $")
        st.plotly_chart(fig,use_container_width=True)
    plot_raw_data()

if processed_data.empty == False:
    def plot_Signal_MACD():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Signal'], name="Signal"))
        fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['MACD'], name="MACD"))
        fig.layout.update(title_text="Signal - MACD Chart", xaxis_rangeslider_visible=True,yaxis_title="USD $")
        st.plotly_chart(fig,use_container_width=True)

    def plot_SMA_EMA():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['Close'], name="stock_close"))
        fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['SMA'], name="SMA"))
        fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['EMA'], name="EMA"))
        fig.layout.update(title_text="SMA - EMA Chart", xaxis_rangeslider_visible=True,yaxis_title="USD $")
        st.plotly_chart(fig,use_container_width=True)
    
    def plot_RSI():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=processed_data['Date'], y=processed_data['RSI'], name="RSI"))
        fig.layout.update(title_text="RSI Chart", xaxis_rangeslider_visible=True,yaxis_title="USD $")
        st.plotly_chart(fig,use_container_width=True)
    
    plot_SMA_EMA()
    plot_Signal_MACD()
    plot_RSI()

def predict_price(selected_stock,data):
    sentiment_analysis_result = sentiment_analyzer.Analysis(selected_stock)
    tomorrow_price, today_predicted_price, today_actual_price = predictor.predict(data,selected_stock)
    return tomorrow_price, today_predicted_price, today_actual_price, sentiment_analysis_result
 
st.write('---')
st.header("Prediction")
if st.button("Predict"):
    loading_message = st.success("Loading Model...")
    if selected_stock != None:
        tomorrow_price, today_predicted_price, today_actual_price, sentiment_analysis_result = predict_price(selected_stock,data)
    loading_message.success("Completed")
    st.markdown(f"""
    ## Today : {TODAY}
    **Predicted Price** : {today_predicted_price} **USD**
    <br>
    **Actual Price** : {today_actual_price} **USD**
    <br>
    <br>
    ## Tomorrow : {(datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")}
    **Predicted Price** : {tomorrow_price} **USD**
    <br>
    ## Current Stock News Sentiment Analysis 
    **Result** : {sentiment_analysis_result}
    """, unsafe_allow_html=True)
else:
    st.write("Click the button above to run the prediction analysis")