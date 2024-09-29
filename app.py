import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from scipy import stats
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
        data['TMRW'] = data['Close'].shift(1)
        data['Label'] = (data['TMRW'] > data['Close']).astype(int)
        TMRW = data['TMRW'].to_list()
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

        model = SVR(C=1.5)
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

        data['Day'] = np.arange(len(data))

        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(data['Day'][-500:], prices[-500:])

        # Prepare predictions
        future_days = 90
        predictions = {
            'Date': [],
            'LinReg_Predicted': []
        }

        # Extend predictions for the next 90 days
        for i in range(1, future_days + 1):
            future_date = data.index[-1] + pd.Timedelta(days=i)
            predictions['Date'].append(future_date)

            # Predict using Linear Regression
            future_day = data['Day'].iloc[-1] + i
            future_price_linreg = slope * future_day + intercept
            predictions['LinReg_Predicted'].append(future_price_linreg)

            # Create DataFrame for predictions
        predictions_df = pd.DataFrame(predictions)

            # Convert dates to pandas datetime
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])[2:]

            # Display predictions

        labels = data['Label'].to_list()
        prices = TMRW
        for i in zip(np.array(predictions_df['LinReg_Predicted'][-90:]),np.array(TMRW[-90:])):
                prices.append(i[0])
                if i[0] > i[1]:
                    labels.append(1)
                else:
                    labels.append(0)
        dates = data.index.to_list() + predictions_df['Date'].to_list()[92:]
        prices = prices[-(len(dates)):]  # Trim prices to match the length of dates
        labels = labels[-(len(dates)):]  # Trim labels to match the length of dates

        a = pd.DataFrame({"Prices":prices,"Label":labels},index=dates)
        st.write()
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
        fig4 = plot_time_series_with_labels(np.array(dates), np.array(prices), np.array(labels))

        # To display the plot (if using Streamlit)
        st.plotly_chart(fig4)

        
            
            
        # Create the Plotly figure for predicted prices with color-coded states
        preds_csv = preds_csv.to_csv(index=False)
        st.download_button(label="Export Predictions as csv",data=preds_csv,file_name="preds.csv",mime="text/csv",)
    except Exception as e:
        st.error("Stock Not Found")
        st.error(e)
