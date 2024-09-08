import streamlit as st
import pandas as pd
import numpy as np
import locale
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import norm
from itertools import combinations


############# GRAFICI PER COMPARATOR ################
def compare_one_metric_reports(data_list, metric):
    fig = px.bar(data_list, y=metric, text=metric, color=metric, color_continuous_scale='RdYlGn')
    fig.update_traces(textposition='outside')
    fig.update_layout(title =f"{metric} vs Systems", xaxis_title="")
    return fig

def fig2_scatter(data_list, metric_one, metric_two):
    fig = px.scatter(data_list, x=metric_one, y=metric_two, 
        color=metric_one, color_continuous_scale='RdYlGn', 
        hover_name=data_list.index, text=data_list.index)
    fig.update_traces(textposition='top center')  # Aggiungi questa riga
    fig.update_layout(xaxis_title=metric_one, yaxis_title=metric_two) 
    return fig

def fig3_bar(data_list, metric_one, metric_two):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data_list.index, y=data_list[metric_one],  name=metric_one))
    fig.add_trace(go.Bar(x=data_list.index, y=data_list[metric_two],  name=metric_two))
    fig.update_layout(xaxis=dict(tickmode='array', tickvals=data_list.index, ticktext=data_list.index), showlegend=True)
    return fig


### EQUITY LINE + DRAWDOWN
def gen_equityDrawdown(df, equity, drawdown, drawdownperc):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True)
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=equity, 
            mode='lines',
            name='Equity',
            fill='tozeroy',
            line=dict(color='green', width=1.5),
            showlegend=False
        ), 
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=drawdown, 
            name='Drawdown $',
            fill='tozeroy', 
            line=dict(color='firebrick', width=1.5),
            showlegend=False
        ), 
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=drawdownperc, 
            name='Drawdown %',
            fill='tozeroy', 
            line=dict(color='firebrick', width=1.5),
            showlegend=False
        ), 
        row=3, col=1
    )
    fig.update_layout( 
        width=1400, 
        height=1200, 
        hovermode='x', 
        yaxis=dict(domain=[0.4, 1]), 
        yaxis2=dict(domain=[0.2, 0.4]),
        yaxis3=dict(domain=[0, 0.2])
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_yaxes(autorange="reversed", row=3, col=1)
    fig.update_xaxes(row=1, col=1, 
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    return fig


### EQUITY LINE SINGOLE X PORTFOLIO
def create_equity_line_plot(df_equity, titolo, yTitolo):
    if isinstance(df_equity, pd.DataFrame):
        if len(df_equity.columns) == 0:
            # Se ci è solo una colonna in un DataFrame, crea un semplice grafico a linea
            col_name = df_equity.columns[0]
            fig = go.Figure(data=go.Scatter(x=df_equity.index, y=df_equity[col_name], mode='lines', name=col_name))
        else:
            # Se ci sono più colonne in un DataFrame, crea un grafico con tracce multiple
            traces = []
            for col in df_equity.columns:
                traces.append(go.Scatter(x=df_equity.index, y=df_equity[col], mode='lines', name=col ))
            fig = go.Figure(data=traces)
    elif isinstance(df_equity, pd.Series):
        # Se è un oggetto Series, crea un semplice grafico a linea
        fig = go.Figure(data=go.Scatter(x=df_equity.index, y=df_equity.values, mode='lines'))
    fig.update_layout(
        title= titolo,
        yaxis_title=yTitolo,
        #hovermode='x',
        width=1400,
        height=800
    )
    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_traces(mode='lines')
    fig.layout.template = 'plotly_dark'
    
    return fig



# 1..12
def portplussingol(str_dfEquity,tot_df, drawdownperc):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.3], vertical_spacing=0)
    for col in str_dfEquity.columns:
        fig.add_trace(go.Scatter(x=str_dfEquity.index, y=str_dfEquity[col], mode='lines', name=col), row=1, col=1)  
    fig.add_trace(go.Scatter(x=tot_df.index, y=tot_df['Cumulative P/L'], mode='lines', name="Portfolio", line=dict(color='green', width=3)), row=1, col=1)
    # Add the drawdown line to a separate subplot
    fig.add_trace(
        go.Scatter(
            x=tot_df.index, 
            y=drawdownperc, 
            name='Drawdown %',
            fill='tozeroy', 
            line=dict(color='firebrick', width=1.5),
            showlegend=False
        ), row=2, col=1
    )
    fig.update_layout(
        yaxis_title='Equity',
        hovermode='x',
        width=1400,
        height=1200
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    return fig


# 1.12
def plot_drawdown_periods(df, drawdown_periods):
    fig = go.Figure()
    # Aggiungi l'equity curve
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative P/L'], mode='lines', name='Equity Curve', line=dict(color='blue')))
    # Evidenzia i periodi di drawdown
    for start, end in drawdown_periods:
        diff_days = (end - start).days
        # Aggiungi un'area riempita per il periodo di drawdown
        fig.add_trace(go.Scatter(
            x=[start, end, end, start, start],
            y=[df.loc[start, 'Cumulative P/L'], df.loc[end, 'Cumulative P/L'], 0, 0, df.loc[start, 'Cumulative P/L']],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='rgba(255, 0, 0, 0)'),
            hoverinfo='text',
            text=f'Drawdown Period: {diff_days} days',
            showlegend=False
        ))
    # Configura il layout del grafico
    fig.update_layout(
        yaxis_title='Cumulative P/L',
        showlegend=False,
        width=1400,
        height=800
    )
    return fig


### METRICHE GENERALI
def metriche_general(st, netProfit, GrossWin, GrossLos, ProfitcFactor, avg, percProfit,
                           returnOnAccount, averageWin, averageLoss, numTrades, total_win_trades,
                           total_loss_trades, max_drawdown, max_drawdownPerc, duration, maxTradeLoss, dd_days):
    
    netProfit = locale.currency(netProfit, grouping=True)
    st.write(":blue[Net Profit: ]", netProfit)
    
    GrossWin = locale.currency(GrossWin, grouping=True)
    st.write(":blue[Gross Win: ]", GrossWin)
    
    GrossLos = locale.currency(GrossLos, grouping=True)
    st.write(":blue[Gross Loss: ]", GrossLos)
    
    st.write(":violet[Profit Factor: ]{:.2f}".format(ProfitcFactor))
    
    avg = locale.currency(avg, grouping=True)
    st.write(":violet[Average Trade: ]", avg)
    
    st.write(":violet[Percent Profitable: ] {:.2f}%".format(percProfit))
    
    st.write(":orange[Return on Account:] {:.0f}".format(returnOnAccount))
    
    averageWin = locale.currency(averageWin, grouping=True)
    st.write(":blue[Average Win: ] ", averageWin)
    
    averageLoss = locale.currency(averageLoss, grouping=True)
    st.write(":blue[Average Loss: ] ", averageLoss)
    
    st.write(":green[Total Trades: ]{:.0f}".format(numTrades))
    st.write(":green[Total Win Trades: ]{:.0f}".format(total_win_trades))
    st.write(":green[Total Loss Trades: ]{:.0f}".format(total_loss_trades))
    st.write("Avg Time in Trade: ", duration)

    max_drawdown = locale.currency(max_drawdown, grouping=True)
    st.write(":red[Max Trade Loss: ] {:.0f}".format(maxTradeLoss))
    st.write(":red[Max Drawdown: ]", max_drawdown)
    st.write(":red[Max Drawdown: ] {:.2f}%".format(max_drawdownPerc))

    st.write(":red[Max Drawdown Days: ] {:.0f}".format(dd_days))



### TAB METRICHE FINALI ULTIME
def metriche_finali(st, ritorni_settimanali, ritorni_mensili, ritorni_annuali,
                           twoMonthPos, newHigh, last_drawdown, actual_percentage):
    
    st.subheader("Last 2 months")
    st.write(twoMonthPos)
    
    st.subheader("New equity Highs")
    st.write(newHigh)

    ritorni_settimanali = locale.currency(ritorni_settimanali, grouping=True)
    st.write(":orange[Average Weekly Returns:]", ritorni_settimanali)
    
    ritorni_mensili = locale.currency(ritorni_mensili, grouping=True)
    st.write(":orange[Average Monthly Returns:]", ritorni_mensili)
    
    ritorni_annuali = locale.currency(ritorni_annuali, grouping=True)
    st.write(":orange[Average Annual Returns:]", ritorni_annuali)
    
    last_drawdown = locale.currency(last_drawdown, grouping=True)
    st.write(":red[Actual drawdown: ]", last_drawdown)
    st.write(":red[Actual drawdown on Max DD %: ]{:.2f}".format(actual_percentage))

##### TRADE ANALYSIS >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# TRADE NEL TEMPO
def tradeTime_graph(df):
    fig = go.Figure(data=[
    go.Bar(
                x=df.index,
                y=df['Profit/Loss'],
                marker_color=df['Color'] 
            )
        ])
    fig.update_layout(title="Profit/Loss in Time")
    return fig

# MAE E MFE
def mae_mfe(tot_df, maeORmfe, title):
    color_map = {'Profit': 'green', 'Loss': 'red'}
    fig = px.scatter(tot_df, x='Profit/Loss', y=maeORmfe, size='MaxPositionProfit',
                color='TradeResult', color_discrete_map=color_map,
                title=title,
                labels={'Profit/Loss': 'Profit/Loss', maeORmfe: maeORmfe},
                hover_data=['MaxPositionProfit', 'MaxPositionLoss'],
                height=800, width=600)
    return fig

def tradeinTime(tot_df):
    color_map = {'Profit': 'green', 'Loss': 'red'}

    tot_df['DaysInTrading'] = (tot_df.index- tot_df['EntryDate-Time']).dt.total_seconds() / 86400
    
    fig = px.scatter(tot_df.sort_values(by="DaysInTrading"), x="DaysInTrading", y="Profit/Loss", 
        color='TradeResult', color_discrete_map=color_map,
        labels={"timeInTrading": "Time in Trading", "Profit/Loss": "Profit/Loss"})

    fig.update_layout(title="Time in Trading vs. Profit/Loss",
                    xaxis_title="Days in Trading",
                    yaxis_title="Profit/Loss")
    return fig
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



###### ALLOCATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### PORTFOLIO OPEN STRATEGIES & MARGINS OVER TIME
def create_overTime(df, yColumn, titleGraph, titleXaxes, titleYaxes):
    fig = px.bar(df, x=df.index, y=yColumn, title=titleGraph, height=500)
    fig.update_xaxes(title_text=titleXaxes)
    fig.update_yaxes(title_text=titleYaxes)
    return fig

### TABELLA RENDIMENTI MENSILI
def rendimenti_mensili(df):
    monthly_returns = pd.pivot_table(df, values='Profit/Loss', index=df.index.year, columns=df.index.month, aggfunc='sum')  
    month_dict = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Otc', 11: 'Nov', 12: 'Dec'}
    monthly_returns = monthly_returns.fillna(0).astype(int).rename(columns=month_dict)
    monthly_returns = monthly_returns.sort_index(ascending=False)
    monthly_returns['YTD'] = monthly_returns.sum(axis=1)  
    
    # Funzione per colorare i mesi in base ai loro rendimenti
    def color_negative_red(val):
        color = 'red' if val < 0 else 'green'
        return 'color: %s' % color
    
    monthly_returns.index.name = "Year"    
    st.dataframe(monthly_returns.style.map(color_negative_red), column_config={"Year": st.column_config.NumberColumn(format="%d")}, use_container_width=True)
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



##### CORRELATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def apply_corr_style(corrMatrix):
    return corrMatrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format(precision=5)

##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



##### DISTRIBUTION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
### DISTRIBUZIONE NORMALE
def plot_normal_distribution(mu, sigma, data, nameCol):
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    y = norm.pdf(x, mu, sigma)
    hist_trace = go.Histogram(x=data[nameCol], nbinsx=50, histnorm='probability density', opacity=0.6, marker=dict(color='blue'), name = 'Distr')
    curve_trace = go.Scatter(x=x, y=y, mode='lines', line=dict(color='red', dash='dash'), name = 'Curve')
    layout = go.Layout(xaxis=dict(title='Profit/Loss'), yaxis=dict(title='Probability Density'), title='Normal Distribution of Profit/Loss')
    fig = go.Figure(data=[hist_trace, curve_trace], layout=layout)
    fig.update_layout(showlegend=False, width=1400, height=700)
    st.plotly_chart(fig, use_container_width=True)
### DISTRIBUZIONE DOPPIA COLONNA
def plot_distr_doppia(df1, col1, color1, df2, col2, color2):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df1[col1], name=col1, marker_color=color1))
    fig.add_trace(go.Histogram(x=df2[col2], name=col2, marker_color=color2))
    fig.update_layout(title=f'Distribution of {col1} and {col2}',
                    xaxis_title='Profit/Loss',
                    yaxis_title='Count',
                    showlegend=False, 
                    width=1400, 
                    height=700)
    st.plotly_chart(fig, use_container_width=True)
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



##### ASSET ALLOCATION >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def plot_assAll(label, symbol, title):
    # Crea il grafico a torta
    fig = go.Figure(data=[go.Pie(labels= label, values=symbol, showlegend=False)])
    # Aggiungi il nome alle etichette del grafico
    fig.update_traces(textposition='inside', textinfo='label+percent')
    fig.update_layout(title_text=title, height=500, width=500)
    st.plotly_chart(fig,  use_container_width=True)
### TABELLA GRAFICO ANNI
def grafico_annuali(annuali, long=None, short=None):
    fig = go.Figure()
    if long is not None and short is not None:
        fig.add_trace(go.Bar(x=annuali.index, y=annuali[long], name='Long',
                        marker=dict(color=['red' if val < 0 else 'green' for val in annuali[long]])))
        fig.add_trace(go.Bar(x=annuali.index, y=annuali[short], name='Short',
                    marker=dict(color=['purple' if val < 0 else 'darkgreen' for val in annuali[short]])))
    if long is None and short is None:
        fig.add_trace(go.Bar(x=annuali.index, y=annuali.values,
                    marker=dict(color=['red' if val < 0 else 'green' for val in annuali.values])))
    
    title = 'Sum of Long/Short per year' if long is not None and short is not None else 'Sum of Profit/Loss per year'
    fig.update_layout(title=title, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
### TABELLA GRAFICO MENSILE
def grafico_mensile(mensile, long=None, short=None):
    fig = go.Figure()
    if long is not None and short is not None:
        fig.add_trace(go.Bar(x=mensile.index, y=mensile[long], name = 'Long',
                    marker_color=mensile[long].apply(lambda x: 'green' if x>=0 else 'red')))
        fig.add_trace(go.Bar(x=mensile.index, y=mensile[short], name = 'Short',
                marker_color=mensile[short].apply(lambda x: 'darkgreen' if x>=0 else 'purple')))
    if long is None and short is None:
        fig.add_trace(go.Bar(x=mensile.index, y=mensile['Profit/Loss'],
            marker_color=mensile['Profit/Loss'].apply(lambda x: 'green' if x>=0 else 'red')))

    fig.update_layout(title='Monthly returns', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
### GRAFICO SETTIMANALE
def grafico_settimanale(settimanale, titolo, long=None, short=None):
    fig = go.Figure()
    if long is not None and short is not None:
        fig.add_trace(go.Bar(x=settimanale.index, y=settimanale[long], 
                    name='Long', marker_color=settimanale[long].apply(lambda x: 'green' if x >= 0 else 'red')))
        fig.add_trace(go.Bar(x=settimanale.index, y=settimanale[short],
                    name='Short', marker_color=settimanale[short].apply(lambda x: 'darkgreen' if x >= 0 else 'purple')))
    if long is None and short is None:
        fig.add_trace(go.Bar(x=settimanale.index, y=settimanale,
                    marker_color=settimanale.apply(lambda x: 'green' if x>=0 else 'red')))
    fig.update_layout(title=titolo, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
### GIORNI DEL MESE
def grafico_ggMese(giornimese, titolo, long=None, short=None):
    fig = go.Figure()
    if long is not None and short is not None:
        fig.add_trace(go.Bar(x=giornimese.index, y=giornimese[long], name='Long',
                            marker=dict(color=['red' if val < 0 else 'green' for val in giornimese[long]])))   
        fig.add_trace(go.Bar(x=giornimese.index, y=giornimese[short], name='Short',
                            marker=dict(color=['purple' if val < 0 else 'darkgreen' for val in giornimese[short]])))
    if long is None and short is None:
        fig.add_trace(go.Bar(x=giornimese.index, y=giornimese.values,
                                marker=dict(color=['red' if val < 0 else 'green' for val in giornimese.values])))
    fig.update_layout(title=titolo, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
### GRAFICO RITORNI ORARI
def grafico_ritorni_orari(ret_hou, titolo, long=None, short=None):
    fig = go.Figure()
    if long is not None and short is not None:
        fig.add_trace(go.Bar(x=ret_hou.index, y=ret_hou[long], name='Long',
                            marker=dict(color=['red' if val < 0 else 'green' for val in ret_hou[long]])))
        fig.add_trace(go.Bar(x=ret_hou.index, y=ret_hou[short], name='Short',
                            marker=dict(color=['purple' if val < 0 else 'darkgreen' for val in ret_hou[short]])))
    if long is None and short is None:
        fig.add_trace(go.Bar(x=ret_hou.index, y=ret_hou.values,
                            marker=dict(color=['red' if val < 0 else 'green' for val in ret_hou.values])))
    fig.update_layout(title=titolo, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




##### EQUITY-CONTROL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def plot_equity_control(df):
    fig = px.line(
    df, 
    x=df.index, 
    y=['Control curve', 'Controlled equity', 'Original Equity'], 
    title="Equity Chart",
    color_discrete_map={
        'Control curve': 'red',
        'Controlled equity': 'blue',
        'Original Equity': 'green'})
    fig.update_layout(
        yaxis_title="Profit",
        xaxis_title="Time",
        hovermode='x',
        showlegend=False, 
        width=1400, 
        height=800,) 
    return fig
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




##### MONTE-CARLO >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""
def plot_Montecarlo(cumulative_returns_transposed):
    colors = ['rgb(0, 0, 255)'] * (len(cumulative_returns_transposed.columns) - 1)  # Colore unico per tutte le altre colonne
    colors.append('rgb(255, 0, 0)')  # Colore vivace per la colonna "Equity"
    fig = go.Figure()
    fig.update_layout(height=800, showlegend=False)

    for i, col in enumerate(cumulative_returns_transposed.columns):
        fig.add_trace(go.Scatter(x=cumulative_returns_transposed.index, y=cumulative_returns_transposed[col],
                                 mode='lines', name=col, line_shape='hv', line=dict(color=colors[i])))
    
"""
def plot_Montecarlo(cumulative_returns_transposed):
    # Scegli un valore adatto per il campionamento (ad esempio, ogni 10° punto dati)
    n = 10
    downsampled_data = cumulative_returns_transposed.iloc[::n]
    colors = ['rgb(0, 0, 255)'] * (len(downsampled_data.columns) - 1)  # Colore unico per tutte le altre colonne
    colors.append('rgb(255, 0, 0)')  # Colore vivace per la colonna "Equity"
    fig = go.Figure()
    fig.update_layout(height=700, showlegend=False)

    for i, col in enumerate(downsampled_data.columns):
        fig.add_trace(go.Scatter(x=downsampled_data.index, y=downsampled_data[col],
                                 mode='lines', name=col, line_shape='hv', line=dict(color=colors[i])))

    return fig
    

def plot_equity(historical_equity_curve, total_simulated_equity, percentile_5, percentile_95, num_future_trades):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(historical_equity_curve))), y=historical_equity_curve, mode='lines', name='Historical Equity'))
    x_simulated = list(range(len(historical_equity_curve), len(historical_equity_curve) + num_future_trades))
    fig.add_trace(go.Scatter(x=x_simulated, y=total_simulated_equity, mode='lines', name='Simulated Equity'))
    # Aggiungi i percentili
    fig.add_trace(go.Scatter(x=x_simulated, y=percentile_5, mode='lines', name='5° Percentile', line=dict(dash='dash')))
    fig.add_trace(go.Scatter(x=x_simulated, y=percentile_95, mode='lines', name='95° Percentile', line=dict(dash='dash')))
    # Imposta le etichette degli assi
    fig.update_xaxes(title_text='Number of Operations', range=[0, len(historical_equity_curve) + num_future_trades])
    fig.update_yaxes(title_text='Equity')
    # Imposta le dimensioni del grafico
    fig.update_layout(width=800, height=800)
    st.plotly_chart(fig, use_container_width=True)
##### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<