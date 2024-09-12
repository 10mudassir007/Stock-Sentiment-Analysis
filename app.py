import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import MinMaxScaler

ticker = st.text_input("Enter text")
button = st.button("Predict")

if button:
    stock = yf.Ticker(ticker)
    data = stock.history(period='5y')
    stock_price = data['Close'][-1]
    if True:
        sc = MinMaxScaler(feature_range=(0, 1))
        data['Close'] = sc.fit_transform(data[['Close']])
        data['Open'] = sc.fit_transform(data[['Open']])
        data['Low'] = sc.fit_transform(data[['Low']])
        data['High'] = sc.fit_transform(data[['High']])
        data['Sum'] = (data['Close'] + data['High'] + data['Low'] + data['Open'])
        
        # Adjusted Sentiments Calculation
        sentiments = [0 for _ in range(len(data['Sum']))]
        
        # Use rolling window to smooth sentiment calculation
        for i in range(30, len(data['Close'])):
            if data['Close'][:i].mean() is not np.nan:
                # Calculate weighted difference between price and sum
                curr_price = (data['Close'][i] + data['High'][i] + data['Low'][i] + data['Open'][i])
                sum_ma = data['Sum'][:i].mean()  # Rolling average of 'Sum'
                
                # Calculate Sentiments as a ratio-based comparison to better align with price
                sentiments[i] = (curr_price / sum_ma) - 1  # Price deviation from sum MA
            else:
                sentiments[i] = 0
        
        data['Price'] = data['Sum'] / 4
    
        sentiment_scaler = MinMaxScaler(feature_range=(0, 1))
        data['Sentiments'] = sentiment_scaler.fit_transform(np.array(sentiments).reshape(-1, 1))
        data['20MA'] = data['Sentiments'].rolling(window=20).mean()
        data['50MA'] = data['Sentiments'].rolling(window=50).mean()
        data['200MA'] = data['Sentiments'].rolling(window=200).mean()

        st.write(f"<h3>Stock Price: {round((stock_price),2)}<h3>",unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index[-20:], y=data['20MA'][-20:], mode='lines', name='20-day MA', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-50:], y=data['50MA'][-50:], mode='lines', name='50-day MA', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-200:], y=data['200MA'][-200:], mode='lines', name='200-day MA', line=dict(color='blue', dash='dash')))

        fig.update_layout(
            title=f'{ticker} Stock Price Over Last Month',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_range=[data.index[-180], data.index[-1]],
            template='plotly_dark'
        )
        st.plotly_chart(fig)
        score, p_value, _ = coint(data['Sum'], data['Sentiments'])

        # Display the p-value from cointegration tes

        # Plot the stock prices and sentiments
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Stock Price Sum', line=dict(color='blue', dash='solid')))
        fig2.add_trace(go.Scatter(x=data.index, y=data['Sentiments'], mode='lines', name='Sentiment', line=dict(color='orange', dash='solid')))

        fig2.update_layout(
            title=f'{ticker} Cointegration Graph (Price vs Sentiment)',
            xaxis_title='Date',
            yaxis_title='Normalized Price / Sentiment',
            xaxis_range=[data.index[-180], data.index[-1]],
            template='plotly_dark'
        )

        # Plot the cointegration graph
        st.plotly_chart(fig2)

    # except Exception as e:
    #     st.error("Stock Not Found")
