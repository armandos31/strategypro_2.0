import pandas as pd
import numpy as np
from datetime import datetime
import os
import random
import streamlit as st

column_names = ['EntryDate-Time', 'EntryPrice', 'ExitDate-Time', 'ExitPrice', 'SymbolName', 'MarketPosition', 'Max Contracts', 'Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss', 'Null']
column_names2 = ['EntryDate-Time', 'EntryPrice', 'ExitDate-Time', 'ExitPrice', 'SymbolName', 'MarketPosition', 'Max Contracts', 'Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss']

### MAIN DF PROCESSION
def prepare_df_strategy_pro(df_report: pd.DataFrame, report_name: str):
    if len(df_report.columns) == 11:
        df_report.columns = column_names
        df_report = df_report.drop(['Null'], axis=1)
    else:
        df_report.columns = column_names2
    df_report['MarketPosition'] = df_report['MarketPosition'].astype('int16') 
    df_report['EntryPrice'] = df_report['EntryPrice'].astype('float32')
    df_report['ExitPrice'] = df_report['ExitPrice'].astype('float32')
    df_report['StrategyName'] = f"{report_name}"
    return df_report

## DF PROCESSING MAIN SINGLE SYSTEM
def prepare_df_comparator(df_report: pd.DataFrame):
    if len(df_report.columns) == 11:
        df_report.columns = column_names
        df_report = df_report.drop(['Null'], axis=1)
    else:
        df_report.columns = column_names2
    df_report = df_report.drop(['EntryDate-Time', 'EntryPrice', 'ExitPrice', 'SymbolName', 'MarketPosition', 'Max Contracts'], axis=1)
    df_report['Profit/Loss'] = df_report['Profit/Loss'].astype('float32')
    df_report['MaxPositionLoss'] = df_report['MaxPositionLoss'].astype('float32')
    df_report['MaxPositionProfit'] = df_report['MaxPositionProfit'].astype('float32')
    df_report['ExitDate-Time'] = pd.to_datetime(df_report['ExitDate-Time'], format="mixed", dayfirst=True)
    df_report = df_report.set_index('ExitDate-Time')
    df_report['Cumulative P/L'] = df_report['Profit/Loss'].cumsum().astype('float32') 
    df_report['Max Equity'] = df_report['Cumulative P/L'].cummax()                   
    df_report['Drawdown']  = df_report['Max Equity'] - df_report['Cumulative P/L']          
    return df_report

### CALCULATION OF NUMBER OF TRADES
def num_Trades(df):
    numTrades = len(df.index)
    return numTrades

### CALCULATION OF NUMBER OF WINNING TRADES
def calc_winTrades(df):
    win_trades = (df['Profit/Loss'] >= 0).sum()
    return win_trades

### CALCULATION OF NUMBER OF LOSING TRADES
def calc_losTrades(df):
    los_trades = (df['Profit/Loss'] < 0).sum()
    return los_trades

### MONTH CALCULATIONS
def creation_Month():
    current_month = datetime.now().month # Current month
    previous_month = (datetime.now() - pd.DateOffset(months=1)).month # Last month
    previous_year = (datetime.now() - pd.DateOffset(months=1)).year # Last year
    two_months_ago_month = (datetime.now() - pd.DateOffset(months=2)).month # Month of 2 month ago
    two_months_ago_year = (datetime.now() - pd.DateOffset(months=2)).year # Year of two month ago
    return current_month, previous_month, previous_year, two_months_ago_month, two_months_ago_year

### CREATE 2M DATAFRAME
def twoMon_df(df, two_months_ago_month, two_months_ago_year, previous_month, previous_year, current_month):
    mask = (df.index.month == two_months_ago_month) & (df.index.year == two_months_ago_year) | (df.index.month == previous_month) & (df.index.year == previous_year) & (df.index.month != current_month)
    month_df = df.loc[mask] # --> 2  DATAFRAME
    return month_df

### SEQUENCE PREPROCESSING
def processDf_consecutive(df):
    filtered_df = df[(df['Profit/Loss'] > 0) | (df['Profit/Loss'] < 0)].copy() 
    filtered_df['Profit/Loss'] = filtered_df['Profit/Loss'].apply(lambda x: 1 if x > 0 else -1) 
    filtered_df = filtered_df[['Profit/Loss']].copy().reset_index() 
    return filtered_df

### POSITIVE SEQUENCE
def positive_sequence(df):
    df['is_positive'] = df['Profit/Loss'] == 1 
    df['group'] = (df['is_positive'] != df['is_positive'].shift()).cumsum() 
    sequence_counts = df[df['is_positive']].groupby('group').size()
    result_df = sequence_counts.value_counts().reset_index() 
    result_df.columns = ['Number of Series', 'Winning Series']
    result_df.sort_values('Number of Series', inplace=True)
    result_df.set_index('Number of Series', inplace=True)
    return result_df

### NEGATIVE SEQUENCE
def negative_sequence(df):
    df['is_negative'] = df['Profit/Loss'] == -1    
    df['group'] = (df['is_negative'] != df['is_negative'].shift()).cumsum() 
    sequence_counts = df[df['is_negative']].groupby('group').size() 
    result_df = sequence_counts.value_counts().reset_index() 
    result_df.columns = ['Number of Series', 'Losing Series']
    result_df.sort_values('Number of Series', inplace=True)
    result_df.set_index('Number of Series', inplace=True)
    return result_df

### CALC DRAWDOWN
def calc_drawdown(equity):
    maxvalue = equity.expanding(0).max()
    drawdown = maxvalue - equity
    return drawdown

### CALC MAX DD %
def calc_maxDDPerc(tot_maxDrawdown, tot_netProfit):
    if tot_netProfit > 0:
        maxDDPerc = (tot_maxDrawdown  / tot_netProfit) * 100
    else:
        maxDDPerc = 0
    return maxDDPerc

### CALC MAX DRAWDOWN
def calc_maxDrawdown(drawdown):
    if drawdown.isnull().all(): max_drawdown = 0
    else: max_drawdown = int(drawdown.max())
    return max_drawdown

### CALC LAST DRAWDOWN
def calc_lastDD(drawdown):
    if len(drawdown) > 0: last_drawdown = int(drawdown.iloc[-1])
    else: last_drawdown = 0
    return last_drawdown

### CALC INCIDENCE MAX DD ON DD 2M & 5%
def calc_inc(last_drawdown, max_drawdown):
    if max_drawdown > 0:
        incidenza = ((last_drawdown/max_drawdown) * 100)
    else:
        incidenza = 0
    return incidenza

### CALC NET PROFIT
def calc_netProfit(df, nomeColonna):
    netProfit = int((df[nomeColonna].sum()))
    return netProfit

### CALC % PROFIT
def calc_percProf(win_trades, numTrades):
    if numTrades == 0: percProfit = 0
    else: percProfit =round(((win_trades * 100) / numTrades) ,2)
    return percProfit

### CALC AVERAGE TRADE
def calc_AvgTrade(netProfit, numTrades):
    if netProfit == 0 or numTrades ==0: avg = 0
    else: avg = round((netProfit / numTrades), 2)
    return avg

### CALC GROSS WIN
def calc_grossWin(df, nomeColonna):
    mask = df[nomeColonna] >= 0
    grossWin = int(df[mask][nomeColonna].sum())
    return grossWin

### CALC GROSS LOSS
def calc_grossLoss(df, nomeColonna):
    mask = df[nomeColonna] < 0
    grossLos = int(df[mask][nomeColonna].sum())
    return grossLos

### CALC PROFICT FACTOR
def calc_profictFactor(GrossWin, GrossLos):
    if GrossLos == 0: ProfitcFactor = round((GrossWin / 1000),2)  
    else: ProfitcFactor = round((GrossWin / (GrossLos * -1)),2)   
    return ProfitcFactor

### CALC AVERAGE WIN
def calc_averageWin(GrossWin, total_win_trades):
    if GrossWin > 0: averageWin = int(GrossWin / total_win_trades)
    else: averageWin = 0
    return averageWin

### CALC AVERAGE LOSS
def calc_averageLoss(GrossLoss, total_loss_trades):
    if GrossLoss < 0: averageLoss = int(GrossLoss / total_loss_trades)  
    else: averageLoss = 0
    return averageLoss


### CALC RETURN ON ACCOUNT
def cacl_retOnAcc(max_drawdown, netProfit):
    accountSize = abs(max_drawdown)
    if netProfit < 0:
        return netProfit  
    elif accountSize == 0:
        return netProfit
    else:
        returnOnAccount = int((netProfit / accountSize) * 100)
    return returnOnAccount

### CALC MAX TRADE LOSS
def calc_maxTradeLoss(df):
    if not df.empty:
        max_trade_loss = int(df[['Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss']].min().min()) 
    else: max_trade_loss = 0 
    return max_trade_loss


### CALC NEW HIGH IN THE LAST MONTH
def check_new_high(equity: pd.DataFrame, previous_month, previous_year, current_month):
    month_df = equity.resample('ME').sum() 
    maskLastMonth = (equity.index.month == previous_month) & \
                    (equity.index.year == previous_year) & \
                    (equity.index.month != current_month)
    lastMonth = equity.loc[maskLastMonth].fillna(0)
    if not lastMonth.isnull().values.any() and lastMonth.max() > 0:
        max_lastEquity = int(lastMonth.max())
    else:
        max_lastEquity = 0
    maskFull = (equity.index.month != previous_month) & \
               (equity.index.month != current_month)
    fullDf = equity.loc[maskFull]
    max_fullEquity = int(fullDf.max())
    newHigh = max_lastEquity >= max_fullEquity
    return newHigh

### CALC LAST 2 MONTH POSITIVE
def check_last_two_months_positive(month_df):
    profit_loss = month_df['Profit/Loss'].resample('ME').sum()
    last_two_months = profit_loss.tail(2) 
    if (last_two_months > 0).all():
        twoMonthPos = True
    else:
        twoMonthPos = False
    return twoMonthPos

### CALC RETURN
def calc_returns(netProfit, corr):
    ritorni = int(netProfit /len(corr))
    return ritorni

### WEEKLY RETURN
def ret_sett(df, sumOrCount, long=None, short=None):
    days = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['weekday'] = df.index.dayofweek
    if long is None and short is None:
        if sumOrCount == 'sum': settimanale = df.groupby('weekday')['Profit/Loss'].sum() 
        else: settimanale = df.groupby('weekday')['Profit/Loss'].count() 
        settimanale.index = settimanale.index.map(days)
    if long is not None and short is not None:
        if sumOrCount == 'sum':
            settimanale_long = df.groupby('weekday')[long].sum() 
            settimanale_short = df.groupby('weekday')[short].sum()
        else:
            settimanale_long = df[df[long] != 0].groupby('weekday')[long].count()
            settimanale_short = df[df[short] != 0].groupby('weekday')[short].count()
        settimanale_long.index = settimanale_long.index.map(days)
        settimanale_short.index = settimanale_short.index.map(days)
        settimanale = pd.DataFrame({1: settimanale_long, -1: settimanale_short})
    return settimanale

### MONTHLY RETURN
def ret_mens(df):
    df = df.resample('ME').sum()
    mesi = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    mensile = df[df.index.month.isin([1,2,3,4,5,6,7,8,9,10,11,12])].groupby(df.index.strftime('%B')).sum()
    mensile = mensile.reindex(mesi, axis=0)
    return mensile

### RETURN DAY OF MONTH
def ret_dayMonth(df, sumOrCount, long=None, short=None):
    if long is None and short is None:
        if sumOrCount == 'sum': giornimese = df.groupby(df.index.day)['Profit/Loss'].sum()
        else: giornimese = df.groupby(df.index.day)['Profit/Loss'].count()
    if long is not None and short is not None:
        if sumOrCount == 'sum':
            giornimeseLong = df.groupby(df.index.day)[long].sum()
            giornimeseShort = df.groupby(df.index.day)[short].sum()
        else:
            giornimeseLong = df.groupby(df.index.day)[long].count()
            giornimeseShort = df.groupby(df.index.day)[short].count()             
        giornimese = pd.DataFrame({1: giornimeseLong, -1: giornimeseShort})
    return giornimese

### YEARLY RETURN
def ret_year(df, long=None, short=None):
    if long is not None and short is not None:
        annualiLong = df.groupby(df.index.year)[long].sum()
        annualiShort = df.groupby(df.index.year)[short].sum()
        annuali = pd.DataFrame({1: annualiLong, -1: annualiShort})
    if long is None and short is None:
        annuali = df.groupby(df.index.year)['Profit/Loss'].sum()
    return annuali

### HOUR RETURN
def ret_hour(df, sumOrCount, long=None, short=None):
    df['Hour'] = df.index.hour
    if long is not None and short is not None:
        if sumOrCount == 'sum':
            long_hourly_summary = df.groupby('Hour')[long].sum()
            short_hourly_summary = df.groupby('Hour')[short].sum()
        else:
            long_hourly_summary = df.groupby('Hour')[long].count()
            short_hourly_summary = df.groupby('Hour')[short].count()
        hourly_summary = pd.DataFrame({1: long_hourly_summary, -1: short_hourly_summary})
    if long is None and short is None:
        if sumOrCount == 'sum': hourly_summary = df.groupby('Hour')['Profit/Loss'].sum()
        else: hourly_summary = df.groupby('Hour')['Profit/Loss'].count()
    return(hourly_summary)

### CALC NORMAL DISTRIBUTION
def calc_normal_distribution(data, nameCol):
    mu = data[nameCol].mean()
    sigma = data[nameCol].std()  
    return mu, sigma

### CALC TRADE DURATION
def duration_Trade(dataframe):
    duration = dataframe['TimeInTrading'].apply(lambda x: pd.to_timedelta(x))
    avg_Duration = duration.mean()
    duration_str = str(avg_Duration)
    duration_without_milliseconds = duration_str.split('.')[0]
    return duration_without_milliseconds

### TOTAL EQUITY MULTI
def total_equity(df):
    df_equity = df
    df_equity = df_equity.apply(lambda x: x.cumsum())
    return df_equity

### ## CALC EQUITY
def calc_equity(df):
    equity = df['Profit/Loss'].cumsum()
    return equity


### NEW DF FOR COL
def divide_tot_df(df, colName):
    if colName == "MarketPosition":
        div_df = df[['MarketPosition', 'Profit/Loss']].pivot_table(index=df.index, columns=colName, values='Profit/Loss', aggfunc='sum', fill_value=0)
    else:
        div_df = pd.pivot_table(df, index=df.index, columns=colName, values='Profit/Loss', aggfunc='sum', fill_value=0)
    return div_df

# LAST DAY TRADE # COMPARATOR
def lastTRDday(df):
    if df.empty:
            return None
    ultima_riga = df.iloc[-1]
    ultima_data_str = ultima_riga.name.strftime('%Y-%m-%d')
    ultima_data = datetime.strptime(ultima_data_str, '%Y-%m-%d').date()
    return ultima_data

# COMPARATOR NETPR/MAXDD
def calc_NetpOnMaxdd(netProfit, max_drawdown):
    if max_drawdown > 0:
        NetpOnMaxdd =  int(netProfit / max_drawdown)
    elif max_drawdown == 0 and netProfit > 0:
        NetpOnMaxdd = 10
    else:
         NetpOnMaxdd = -10
    return NetpOnMaxdd

### ALLOCATION
def calc_assAll(df, colName, sumColName):
    allocation = df.groupby(colName)[sumColName].sum().reset_index()
    return allocation

### DRAWDOWN PERIODS
def calc_drawdown_periods(df):
    if not df.empty:
        drawdown_periods = []
        longest_drawdown_days = 0
        start_date = None

        for date, row in df.iterrows():
            if row['Drawdown'] > 0:  
                if start_date is None:
                    start_date = date
            else:  
                if start_date is not None and date != start_date:
                    end_date = date
                    drawdown_periods.append((start_date, end_date))
                    start_date = None

        # Management of any drawdown in progress at the end of the historical series
        if start_date is not None and df.index[-1] != start_date:
            drawdown_periods.append((start_date, df.index[-1]))


        drawdown_periods = [(start, end) for start, end in drawdown_periods if (end - start).days > 10]
        # Find the longest drawdown period
        if drawdown_periods:
            longest_drawdown = max(drawdown_periods, key=lambda x: x[1] - x[0])
            longest_drawdown_days = (longest_drawdown[1] - longest_drawdown[0]).days
            #print(f"Il più lungo drawdown va da {longest_drawdown[0]} a {longest_drawdown[1]}, con una durata di {(longest_drawdown[1] - longest_drawdown[0]).days} giorni.")
        #else:
        #   print("Non ci sono periodi di drawdown.")

        # Visualizza i drawdown
        #print("Dettagli dei periodi di drawdown:")
        #for start, end in drawdown_periods:  
        #    print(f"Da {start} a {end}, durata: {(end - start).days} giorni.")
    else:
        drawdown_periods = 0
        longest_drawdown_days = 0


    return drawdown_periods, longest_drawdown_days


##### MONTE CARLO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### Randomize and create random returns
def monte_carlo_randomized(df, num_simulations, fun_montecarlo):
    mean_profit_loss = df['Profit/Loss'].mean()
    std_dev_profit_loss = df['Profit/Loss'].std()
    monte_carlo_results = []

    for _ in range(num_simulations):
        if fun_montecarlo == "Randomized Only":
            randomized_operations = df.sample(frac=1)['Profit/Loss'].tolist()
            monte_carlo_results.append(randomized_operations)
        else:
            daily_returns = []
            for _ in range(len(df)):
                daily_return = random.normalvariate(mean_profit_loss, std_dev_profit_loss)
                daily_returns.append(daily_return)
            monte_carlo_results.append(daily_returns)

    df_results = pd.DataFrame(monte_carlo_results)
    cumulative_returns = df_results.cumsum(axis=1)

    df_results_transposed = df_results.transpose()
    cumulative_returns_transposed = cumulative_returns.transpose()
    return cumulative_returns_transposed, df_results_transposed

### CALC METRICHE
def calculate_metrics_simulate(df, df_profitloss, initial_capital):
    results = {}  
    for column in df.columns:
        drawdown = calc_drawdown(df[column])
        max_drawdown = calc_maxDrawdown(drawdown)
        max_dd_perc = round(drawdown / initial_capital * 100, 2)
        maxDrawdownPerc = max_dd_perc.max() 
        netProfit = int((df_profitloss[column].sum()))
        ret_acc = cacl_retOnAcc(max_drawdown, netProfit)
        results[column] = {
            'MaxDrawdown': max_drawdown,
            'Perc Drawdown': maxDrawdownPerc,
            'Net Profit': netProfit,
            'Return On Acc': ret_acc,
        }
    results_df = pd.DataFrame(results).T
    return results_df

### CALC SIMULATED EQUITY
def calc_equity_simul(df, num_simulations, num_future_trades):
    historical_trades = df['Profit/Loss'].tolist()
    historical_equity_curve = np.cumsum(historical_trades)
    simulated_equity_curves = []

    for _ in range(num_simulations):
        selected_trades = np.random.choice(historical_trades, num_future_trades)
        simulated_equity_curve = np.cumsum(selected_trades)
        simulated_equity_curve += historical_equity_curve[-1]  
        simulated_equity_curves.append(simulated_equity_curve)
    percentile_5 = np.percentile(simulated_equity_curves, 5, axis=0)
    percentile_95 = np.percentile(simulated_equity_curves, 95, axis=0)
    total_simulated_equity = []
    for simulated_curve in simulated_equity_curves:
        total_simulated_equity.extend(simulated_curve)
    return historical_equity_curve, total_simulated_equity, percentile_5, percentile_95

### METRICS DISTRIBUTION
def distribution_Metrics(df, colName):
    distribution = df[colName].value_counts().reset_index()
    distribution.columns = [colName, 'Frequency']
    return distribution

##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



##### CORRELATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def create_corr_df(df):
    corr_df = df.drop(['Profit/Loss'], axis=1)
    return corr_df

def rolling_correlation(df, combinazioni_colonne, period):
    df_corr = pd.DataFrame()
    df_avg_corr = pd.DataFrame()
    for coppia in combinazioni_colonne:
        colonna1, colonna2 = coppia

        correlazione = df[colonna1].rolling(window=period).corr(df[colonna2])
        
        colName = f'{colonna1} vs {colonna2}'
        df_corr[colName] = correlazione
        df_corr.fillna(0, inplace=True)

        media = correlazione.mean()
        df_avg_corr.loc[colonna1, colonna2] = media

    return df_corr, df_avg_corr

# Function to replace values ​​less than zero with 0
def replace_values(df, replace_negatives):
    if replace_negatives:
        return df.applymap(lambda x: x if x >= 0 else 0)
    else:
        return df.applymap(lambda x: x if x <= 0 else 0)
        
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<








##### FOLIO MANAGER >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<