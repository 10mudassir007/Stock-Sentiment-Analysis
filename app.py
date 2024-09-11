import numpy as np
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

ticker = st.text_input("Enter text")
button = st.button("Predict")


if button:
    stock = yf.Ticker(ticker)
    data = stock.history(period='5y')
    try:
        data['Sum'] = (data['Close'] + data['High'] + data['Low'] + data['Open'])
        mean_price = data['Sum'][:30].mean()
        
        current_price = data['Close'][-1] + data['High'][-1] + data['Low'][-1] + data['Open'][-1]
        
        sentiments = [0 for _ in range(len(data['Sum']))]
        for i in range(30,len(data['Sum'])):
            if data['Sum'][:i].mean() is not np.nan:
                curr_price = (data['Sum'][i])
                me_price = data['Sum'][:i].mean()
                if curr_price >= me_price:
                    sentiments[i] = (curr_price - me_price)
                elif me_price > curr_price:
                    sentiments[i] = (me_price - curr_price)
            else:
                sentiments[i] = 0
        
        data['Sentiments'] = sentiments
        data['20MA'] = data['Sentiments'].rolling(window=20).mean()
        data['50MA'] = data['Sentiments'].rolling(window=50).mean()
        data['200MA'] = data['Sentiments'].rolling(window=200).mean()
        
        st.write(f"<h3>Stock Price: {round((data['Close'][-1]),2)}<h3>",unsafe_allow_html=True)
        if current_price > mean_price:
            pred = "Bullish"
        elif current_price < mean_price:
            pred = "Bearish"
        else:
            pred = "Neutral"
        st.write(f"<h3>Sentiment:{pred}<h3>",unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index[-20:], y=data['20MA'][-20:], mode='lines', name='20-day MA', line=dict(color='purple', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-50:], y=data['50MA'][-50:], mode='lines', name='50-day MA', line=dict(color='yellow', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-200:], y=data['200MA'][-200:], mode='lines', name='200-day MA', line=dict(color='pink', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-30:], y=data['Sentiments'][-30:], mode='lines', name='Sentiment', line=dict(color='red', dash='dash')))

        fig.update_layout(
                        title=f'{ticker} Stock Price Over Last Month',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        xaxis_range=[data.index[-30], data.index[-1]],
                        template='plotly_dark'
                    )
        st.plotly_chart(fig)
    except Exception as e:
        st.error("Stock Not Found")
