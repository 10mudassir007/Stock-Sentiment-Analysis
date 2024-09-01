import yfinance as yf
import re
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pickle
from tensorflow import keras
import contractions
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

model1 = keras.models.load_model("model2.keras")
model2 = keras.models.load_model("model3.keras")
model3 = keras.models.load_model("model4.keras")
with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)


def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = contractions.fix(text)
    text = ''.join([i for i in text if not i.isdigit()])
    text = ' '.join(text.split())
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text



# Define the text and stock dictionary

stock_dict = {
    "Apple Inc.": "AAPL",
    "Microsoft Corporation": "MSFT",
    "Alphabet Inc.": "GOOGL",
    "Google": "GOOGL",
    "Facebook": "META",
    "Amazon.com, Inc.": "AMZN",
    "Tesla, Inc.": "TSLA",
    "NVIDIA Corporation": "NVDA",
    "Berkshire Hathaway Inc.": "BRK-B",
    "Meta Platforms, Inc.": "META",
    "Johnson & Johnson": "JNJ",
    "Visa Inc.": "V",
    "Walmart Inc.": "WMT",
    "UnitedHealth Group Incorporated": "UNH",
    "Procter & Gamble Co.": "PG",
    "JPMorgan Chase & Co.": "JPM",
    "Mastercard Incorporated": "MA",
    "Exxon Mobil Corporation": "XOM",
    "Bank of America Corporation": "BAC",
    "The Walt Disney Company": "DIS",
    "Intel Corporation": "INTC",
    "Cisco Systems, Inc.": "CSCO",
    "Verizon Communications Inc.": "VZ",
    "Pfizer Inc.": "PFE",
    "Adobe Inc.": "ADBE",
    "Salesforce, Inc.": "CRM",
    "AbbVie Inc.": "ABBV",
    "Netflix, Inc.": "NFLX",
    "Oracle Corporation": "ORCL",
    "AT&T Inc.": "T",
    "Chevron Corporation": "CVX",
    "PepsiCo, Inc.": "PEP",
    "Coca-Cola Company": "KO",
    "Merck & Co., Inc.": "MRK",
    "McDonald's Corporation": "MCD",
    "Broadcom Inc.": "AVGO",
    "IBM Corporation": "IBM",
    "Qualcomm Incorporated": "QCOM",
    "Thermo Fisher Scientific Inc.": "TMO",
    "Honeywell International Inc.": "HON",
    "Roche Holding AG": "RHHBY",
    "Lowe's Companies, Inc.": "LOW",
    "Nike, Inc.": "NKE",
    "Texas Instruments Incorporated": "TXN",
    "Starbucks Corporation": "SBUX",
    "General Electric Company": "GE",
    "United Parcel Service, Inc.": "UPS",
    "PayPal Holdings, Inc.": "PYPL",
    "Bristol-Myers Squibb Company": "BMY",
    "T-Mobile US, Inc.": "TMUS",
    "S&P Global Inc.": "SPGI",
    "Wells Fargo & Company": "WFC",
    "Morgan Stanley": "MS",
    "Royal Dutch Shell plc": "SHEL",
    "Caterpillar Inc.": "CAT",
    "3M Company": "MMM",
    "Goldman Sachs Group, Inc.": "GS",
    "Amgen Inc.": "AMGN",
    "Johnson Controls International plc": "JCI",
    "Eli Lilly and Company": "LLY",
    "Uber Technologies, Inc.": "UBER",
    "American Tower Corporation": "AMT",
    "Lockheed Martin Corporation": "LMT",
    "Colgate-Palmolive Company": "CL",
    "Mondelez International, Inc.": "MDLZ",
    "Gilead Sciences, Inc.": "GILD",
    "Lululemon Athletica Inc.": "LULU",
    "Danaher Corporation": "DHR",
    "Etsy, Inc.": "ETSY",
    "ServiceNow, Inc.": "NOW",
    "Zoom Video Communications, Inc.": "ZM",
    "Square, Inc.": "SQ",
    "Peloton Interactive, Inc.": "PTON",
    "Snap Inc.": "SNAP",
    "Twilio Inc.": "TWLO",
    "Workday, Inc.": "WDAY",
    "Shopify Inc.": "SHOP",
    "Snowflake Inc.": "SNOW",
    "Palantir Technologies Inc.": "PLTR",
    "Datadog, Inc.": "DDOG",
    "NIO Inc.": "NIO",
    "Plug Power Inc.": "PLUG",
    "CrowdStrike Holdings, Inc.": "CRWD",
    "MongoDB, Inc.": "MDB",
    "ChargePoint Holdings Inc.": "CHPT",
    "Li Auto Inc.": "LI",
    "DraftKings Inc.": "DKNG",
    "Robinhood Markets, Inc.": "HOOD",
    "DoorDash, Inc.": "DASH",
    "Beyond Meat, Inc.": "BYND",
    "QuantumScape Corporation": "QS",
    "UiPath Inc.": "PATH",
    "Sea Limited": "SE",
    "Baidu Inc.": "BIDU"
}

txt = st.text_input("Enter text")
button = st.button("Predict")

# Normalize and split the text
if button:
    t1 = [txt]
    text_sq = tokenizer.texts_to_sequences(t1)
    text_f = keras.preprocessing.sequence.pad_sequences(text_sq, maxlen=120)
    pred1 = np.argmax(model1.predict(text_f))
    pred2 = np.argmax(model2.predict(text_f))
    pred3 = np.argmax(model3.predict(text_f))
    if pred1 != pred2 and pred1 != pred3 and pred2 != pred3:
        pred = "Neutral"
    else:
        pred = np.array([pred1,pred2,pred3])
        unique, counts = np.unique(pred, return_counts=True)

# Find the index of the maximum count
        max_count_index = np.argmax(counts)

        # Find the most frequent value
        pred = unique[max_count_index]
        if pred == 0:
            pred = "Negative"
        elif pred == 1:
            pred = "Neutral"
        elif pred == 2:
            pred = "Positive"
    st.write(f"<h3>Sentiment:{pred}<h3>",unsafe_allow_html=True)
    text = txt
    text_normalized = re.sub(r'[^\w\s]', '', text).lower()
    text_split = [word.rstrip('s') for word in text_normalized.split()]

    # Hardcoded cases
    if "tesla" in text_normalized:
        symbol = "TSLA"
        found_name = "Tesla, Inc."
    elif "google" in text_normalized:
        symbol = "GOOGL"
        found_name = "Alphabet Inc."
    elif "meta" in text_normalized or "facebook" in text_normalized:
        symbol = "META"
        found_name = "Meta Platforms, Inc."
    elif "microsoft" in text_normalized:
        symbol = "MSFT"
        found_name = "Microsoft Corporation"
    elif "amazon" in text_normalized:
        symbol = "AMZN"
        found_name = "Amazon"
    else:
        # General matching for other stocks
        symbol = None
        found_name = None
        for name, ticker in stock_dict.items():
            if name.lower() in text_normalized or any(keyword.lower() in text_split for keyword in name.split()):
                symbol = ticker
                found_name = name
                break

        # If no name match is found, check if any stock symbol matches the text
        if not symbol:
            for name, ticker in stock_dict.items():
                if ticker.lower() in text_normalized:
                    symbol = ticker
                    found_name = name
                    break

    # Streamlit display
    if symbol:
        st.write(f"<h3>Stock: {symbol}</h3>", unsafe_allow_html=True)

        # Fetch stock price using yfinance
        stock = yf.Ticker(symbol)
        data = stock.history(period='1mo')

        if not data.empty:
            # Extract data for plotting
            dates = data.index
            close_prices = data['Close']

            # Create Plotly figure
            fig = go.Figure()

            # Add traces
            fig.add_trace(go.Scatter(x=dates, y=close_prices, mode='lines+markers', name='Close Price', line=dict(color='green')))
            fig.add_trace(go.Scatter(x=dates, y=data['Open'], mode='lines+markers', name='Open Price', line=dict(color='red')))
            fig.add_trace(go.Scatter(x=dates, y=data['Low'], mode='lines+markers', name='Low Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=dates, y=data['High'], mode='lines+markers', name='High Price', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=dates, y=data['Close'].rolling(window=5).mean(), mode='lines', name='5-day MA', line=dict(color='purple', dash='dash')))
            fig.add_trace(go.Scatter(x=dates, y=data['Close'].rolling(window=10).mean(), mode='lines', name='10-day MA', line=dict(color='yellow', dash='dash')))
            fig.add_trace(go.Scatter(x=dates, y=data['Close'].rolling(window=20).mean(), mode='lines', name='20-day MA', line=dict(color='pink', dash='dash')))

            # Update layout
            fig.update_layout(
                title=f'{found_name} Stock Price Over Last Month',
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_dark'
            )

            # Show the plot
            st.plotly_chart(fig)
        else:
            st.write('No stock data available.')
    else:
        st.write('No relevant stock symbol found.')
