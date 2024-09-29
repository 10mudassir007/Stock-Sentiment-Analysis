import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from hmmlearn.hmm import *
import plotly.graph_objects as go
#from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


ticker = st.text_input("Enter text")
button = st.button("Predict")

if button:
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='max')
        stock_price = data['Close'][-1]
        prices = data['Close']
        sc = MinMaxScaler(feature_range=(0, 1))
        data['Close'] = sc.fit_transform(data[['Close']])
        data['Open'] = sc.fit_transform(data[['Open']])
        data['Low'] = sc.fit_transform(data[['Low']])
        data['High'] = sc.fit_transform(data[['High']])
        data['Sum'] = (data['Close'] + data['High'] + data['Low'] + data['Open'])
        data['TMRW'] = data['Close'].shift(-1)
        data['Label'] = (data['Close'] < data['TMRW']).astype(int)
        st.write(data)
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

       # sentiment_scaler = MinMaxScaler(feature_range=(min(data['Close']), ))
        data['Sentiments'] = sentiments #sentiment_scaler.fit_transform(np.array(sentiments).reshape(-1, 1))
        data['20MA'] = data['Sentiments'].rolling(window=20).mean()
        data['50MA'] = data['Sentiments'].rolling(window=50).mean()
        data['200MA'] = data['Sentiments'].rolling(window=200).mean()

        current_price = (data['Close'][-1] + data['High'][-1] + data['Low'][-1] + data['Open'][-1])
        mean_price = data['Sum'][:30].mean()

        if current_price > mean_price:
            pred = "Bullish"
        elif current_price < mean_price:
            pred = "Bearish"
        else:
            pred = "Neutral"


        st.write(f"<h3>Sentiment: {pred}<h3>",unsafe_allow_html=True)
        

        st.write(f"<h3>Stock Price: {round((stock_price),2)}<h3>",unsafe_allow_html=True)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=data.index[-20:], y=data['20MA'][-20:], mode='lines', name='20-day MA', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-50:], y=data['50MA'][-50:], mode='lines', name='50-day MA', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=data.index[-200:], y=data['200MA'][-200:], mode='lines', name='200-day MA', line=dict(color='blue', dash='dash')))

        fig.update_layout(
            title=f'{ticker} Stock MAs Over six Month',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_range=[data.index[-180], data.index[-1]],
            template='plotly_dark'
        )
        st.plotly_chart(fig)
        
        dates = np.array(pd.to_datetime(data.index).map(pd.Timestamp.toordinal)).reshape(-1,1)
        today = dates.max()
        future_dates = np.array([today + i for i in range(1,91)]).reshape(-1,1)

        model = SVR(C=100)
        model.fit(dates,prices)

        preds = model.predict(future_dates)

        future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(date)) for date in future_dates.ravel()])
        preds_csv = pd.DataFrame(preds,columns=["Predictions"])
        preds_csv.index = future_dates
        
        past_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(date)) for date in dates[-90:].ravel()])
        #future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(date)) for date in future_dates.ravel()])

        # Concatenate past and future dates
        dates_combined = np.concatenate([past_dates, future_dates])

        # Concatenate the corresponding prices and predictions
        prices_combined = np.concatenate([prices[-90:], preds])
        

        fig1 = go.Figure()
        
        
        fig1.add_trace(go.Scatter(x=dates_combined, y=prices_combined, mode='lines', name='Prices', line=dict(color='blue', dash='solid')))

        fig1.update_layout(
            title=f'{ticker} Stock Price in the next 90 days',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_range=[dates_combined[-180],dates_combined[-1]],
            template='seaborn'
        )

        st.plotly_chart(fig1)

        #prices = np.array(data['Close'])

        model = GaussianHMM(
            n_components=3,
            covariance_type="tied",
            n_iter=10000,
            tol=1e-5,
                             
            random_state=42,   
            
)
        
        prices = np.array(prices).reshape(-1,1)
        model.fit(prices)

        n_steps = len(future_dates)
        future_prices, future_states = model.sample(n_steps)
        prices_combined2 = np.concatenate([prices[-90:],future_prices])

        states_combined = np.concatenate([np.zeros(90), future_states])  # Assume all past prices are state 0 for simplicity
        st.write(states_combined)
        # Create the Plotly figure for predicted prices with color-coded states
        
        fig3 = go.Figure()
        color_map = {
            0: "green",   # Example: State 0 -> Bullish
            1: "orange",  # Example: State 1 -> Neutral
            2: "red"      # Example: State 2 -> Bearish
        }
        # Plot by state, assigning colors to each segment according to the state
        for i in range(1, len(dates_combined)):
            fig3.add_trace(go.Scatter(
                x=dates_combined[i-1:i+1].flatten(),  # Two consecutive dates to form a line segment
                y=prices_combined2[i-1:i+1].flatten(),  # Two consecutive prices
                mode='lines',
                line=dict(color=color_map[future_states[i-90]] if i >= 90 else color_map[0], width=2),
                showlegend=False  # We only want to show one legend entry
            ))

        # Update the layout for better presentation
        fig3.update_layout(
            title=f'{ticker} Stock Price Predictions for the Next 90 Days',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_range=[dates_combined[-180], dates_combined[-1]],  # Show the last 180 days
            template='seaborn'
        )

        # Show the chart in the Streamlit app
        st.plotly_chart(fig3)
        #sents = data['Sentiments'].to_frame()
        #sent_csv = sents.to_csv(index=False)
        def plot_time_series_with_labels(dates, prices, labels):
            fig4 = go.Figure()

            # Define the color map for labels
            color_map = {
                0: "red",    # Bearish (label 0)
                1: "green"   # Bullish (label 1)
            }

            # Plot the time series with different colors based on the labels
            for i in range(1, len(dates)):
                fig4.add_trace(go.Scatter(
                    x=dates[i-1:i+1].flatten(),  # Two consecutive dates to form a line segment
                    y=prices[i-1:i+1].flatten(),  # Two consecutive prices
                    mode='lines',
                    line=dict(color=color_map[labels[i]], width=2),  # Use the label to set the color
                    showlegend=False  # Don't show multiple legend entries for each line segment
                ))

            # Update the layout for better presentation
            fig4.update_layout(
                title='Stock Price with Buy/Sell Labels',
                xaxis_title='Date',
                yaxis_title='Price',
                template='seaborn'
            )

            return fig4

        # Example usage with your data (replace these with actual values)
        dates_combined = np.array(data.index)
        prices_combined2 = np.array(data['Close'])
        labels = np.array(data['Label'])

        # Call the function to create the plot
        fig4 = plot_time_series_with_labels(dates_combined, prices_combined2, labels)

        # To display the plot (if using Streamlit)
        st.plotly_chart(fig4)
        preds_csv = preds_csv.to_csv(index=False)
        st.download_button(label="Export Predictions as csv",data=preds_csv,file_name="preds.csv",mime="text/csv",)
    except Exception as e:
        st.error("Stock Not Found")
        st.error(e)
