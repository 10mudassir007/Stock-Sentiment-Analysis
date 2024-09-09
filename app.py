import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

ticker = st.text_input("Enter text")
button = st.button("Predict")

stock = yf.Ticker(ticker)
data = stock.history(period='1y')
if button:
    try:
        data['20MA'] = data['Close'].rolling(window=20).mean()
        data['50MA'] = data['Close'].rolling(window=50).mean()
        data['200MA'] = data['Close'].rolling(window=200).mean()

        data['Sum'] = (data['Close'][-30:] + data['High'][-30:] + data['Low'][-30:] + data['Open'][-30:] + data['20MA'][-30:] + data['50MA'][-30:] + data['200MA'][-30:])
        mean_price = data['Sum'].mean()
        current_price = data['Close'][-1] + data['High'][-1] + data['Low'][-1] + data['Open'][-1] + data['20MA'][-1] + data['50MA'][-1] + data['200MA'][-1]

        if current_price > mean_price:
            pred = "Bullish"
        elif current_price < mean_price:
            pred = "Bearish"
        else:
            pred = "Neutral"
        st.write(f"<h3>Sentiment:{pred}<h3>",unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index[-30:], y=data['Close'][-30:], mode='lines+markers', name='Close Price', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=data.index[-30:], y=data['Open'][-30:], mode='lines+markers', name='Open Price', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index[-30:], y=data['Low'][-30:], mode='lines+markers', name='Low Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index[-30:], y=data['High'][-30:], mode='lines+markers', name='High Price', line=dict(color='orange')))

        fig.add_trace(go.Scatter(x=data.index[-20:], y=data['20MA'][-20:], mode='lines', name='20-day MA', line=dict(color='purple', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-50:], y=data['50MA'][-50:], mode='lines', name='50-day MA', line=dict(color='yellow', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-200:], y=data['200MA'][-200:], mode='lines', name='200-day MA', line=dict(color='pink', dash='dash')))

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
