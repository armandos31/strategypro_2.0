import streamlit as st
from streamlit_option_menu import option_menu
from MyDB import MyDB
from function import *
from graphs import compare_one_metric_reports, fig2_scatter, fig3_bar

def app():
    if 'isExpandedC' not in st.session_state:
        st.session_state['isExpandedC'] = True
    
    myDB = MyDB('armandinodinodello')

    with st.expander("Comparator Dashboard", expanded=st.session_state['isExpandedC']):
        if 'df_report_not_selected_comparator' not in st.session_state:
            tag_file_content = myDB.get_tag_file()
            reports_names = [entry['file_name'] for entry in tag_file_content]
            reports_tags = [" ".join(entry['tags']) for entry in tag_file_content]
            st.session_state['df_report_not_selected_comparator'] = pd.DataFrame({"Report Names":reports_names, "Tags":reports_tags})

        if 'df_report_selected_comparator' not in st.session_state:
            st.session_state['df_report_selected_comparator'] = pd.DataFrame({"Report Names": [], "Tags": []})

        # Search Bar & Radio Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        col1, col2 = st.columns(2)
        with col1:
            cerca = st.multiselect('Select the reports to analyze', st.session_state['df_report_not_selected_comparator'])
            query = st.text_input("Select Report to Compare")
            genre = st.radio( "FIlter bye", ["Name", "Tag"], index=0)    
            # Listener Search Bar ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # & All Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if query:
                filter = 'Report Names' if (genre == 'Name') else 'Tags'
                df_filtered = st.session_state['df_report_not_selected_comparator'][st.session_state['df_report_not_selected_comparator'][filter].str.contains(query, case=False)]
                st.dataframe(df_filtered)
            elif cerca:
                df_filtered = st.session_state['df_report_not_selected_comparator'][st.session_state['df_report_not_selected_comparator']['Report Names'].isin(cerca)]
                st.dataframe(df_filtered)
            else:
                df_filtered = st.session_state['df_report_not_selected_comparator']
                st.dataframe(st.session_state['df_report_not_selected_comparator'])

            # Select Report Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if st.button("Select", use_container_width=True):
                st.session_state['df_report_selected_comparator'] = pd.concat([st.session_state['df_report_selected_comparator'], df_filtered], axis=0)
                st.session_state['df_report_not_selected_comparator'] = pd.merge(st.session_state['df_report_not_selected_comparator'], df_filtered, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                st.session_state['isExpandedC'] = True
                st.rerun()

        # Selected Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with col2:
            st.text("Selected Reports")
            st.dataframe(st.session_state['df_report_selected_comparator'])
            pul1, pul2 = st.columns(2)
            with pul1:
                if st.button('Empty', use_container_width=True):
                    del st.session_state['df_report_selected_comparator']
                    del st.session_state['df_report_not_selected_comparator']
                    if 'df_report_to_compare' in st.session_state:
                        del st.session_state['df_report_to_compare']
                    st.rerun()

            with pul2:
                if st.button('Compare', use_container_width=True):
                    if not st.session_state['df_report_selected_comparator'].empty:
                        st.session_state['df_report_to_compare'] = st.session_state['df_report_selected_comparator']['Report Names'].to_numpy()
                        st.session_state['isExpandedC'] = False
                        st.rerun()
                    else:
                        st.warning('Error!')

##### SECTION TO COMPARE REPORT
    if 'df_report_to_compare' in st.session_state and len(st.session_state['df_report_to_compare']) > 0:

        current_month, previous_month, previous_year, two_months_ago_month, two_months_ago_year = creation_Month()
        lista = pd.DataFrame()

        menu = option_menu(
            menu_title=None,  # required
            options=["Total", "Last 5% of trades", "Last 2 months", "Last month"],
            icons=[" ", " ", " "," "],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
            orientation="horizontal",
        )
    
        for report_name in st.session_state['df_report_to_compare']:
            df_raw_report = myDB.get_df_report(report_name)
            singole = prepare_df_comparator(df_raw_report)
            singole['Cumulative P/L'] = singole['Profit/Loss'].cumsum().astype('float32') 
            month_df = twoMon_df(singole, two_months_ago_month, two_months_ago_year, previous_month, previous_year, current_month)

            # DataFrame for Calc and Metrics
            if menu == "Total":                                 # 1 DF Total
                df = singole
            elif menu == "Last 5% of trades":                   # 2 DF 5%
                numTrades = num_Trades(singole)
                perc_numTrade = int(numTrades * 5 / 100)
                df = singole[-perc_numTrade:]
            elif menu == "Last 2 months":                       #3 DF Last 2M
                df = month_df
            elif menu == "Last month":                          #3 DF Last Month
                mask = (singole.index.month == previous_month)  & (singole.index.year == previous_year)
                last_month_df = singole.loc[mask]
                df = last_month_df

            GrossWin = calc_grossWin(df, 'Profit/Loss')
            GrossLos = calc_grossLoss(df, 'Profit/Loss')
            win_trades = calc_winTrades(df)
            
            # METRICS
            netProfit = calc_netProfit(df, 'Profit/Loss')                                                       # 1 Metric NET PROFIT
            numTrades = num_Trades(df)                                                                          # 2 Metric NUM TRADE
            avg = int(calc_AvgTrade(netProfit, numTrades))                                                      # 3 Metric Avg Trade (AVG)
            percProfit =int(calc_percProf(win_trades, numTrades))                                               # 4 Metric PERCENT PROFIT
            ProfitcFactor = calc_profictFactor(GrossWin, GrossLos)                                              # 5 Metric PROFICT FACTOR
            all_positive = check_last_two_months_positive(month_df)                                             # 6 Metric LAST 2 MONTH POSITIVE
            newHigh = check_new_high(singole['Cumulative P/L'] , previous_month, previous_year, current_month)  # 7 Metric NEW HIGH IN LAST MONTH
            max_drawdown = calc_maxDrawdown(df['Drawdown'])                                                     # 8 Metric MAX DRAWDOWN
            lastDD = calc_lastDD(df['Drawdown'])                                                                # 9 Metric LAST DRAWDOWN
            inc_DD = int((calc_inc(lastDD, max_drawdown)))                                                      # 10 Metric DD ON MAXX DD
            netpmaxdd = calc_NetpOnMaxdd(netProfit, max_drawdown)                                               # 11 Metric NETPROFIT ON MAXDRAWDOWN
            returnOnAccount = cacl_retOnAcc(max_drawdown, netProfit)                                            # 12 Metric RETURN ON ACCOUNT
            max_trade_loss = calc_maxTradeLoss(df)                                                              # 13 Metric MAX TRADE LOSS
            ultima_data = lastTRDday(df)                                                                        # 14 Metric LAST TRADE DAY
            drawdown_periods, longest_drawdown = calc_drawdown_periods(df)                                      # 15 Metric FIND BEST DRAWDOWN PERIODS
    
            # EQUITY FOR SELECTED DATAFRAME
            new_dataframe = pd.DataFrame({'Profit/Loss': df['Cumulative P/L']})
            data_df = pd.DataFrame({"Profit/Loss": [new_dataframe['Profit/Loss'].tolist()]}) 
        
            # MAKE DF
            data = {
                'Net Profit': [netProfit],
                'Num Trade':[numTrades],
                'Avg Trade': [avg],
                '% Profit': [percProfit],
                'Profit Factor': [ProfitcFactor],
                '2 Mon Prof': [all_positive],
                'New Highs': [newHigh],
                'Max Drawdown': [max_drawdown],
                'Last DD':[lastDD],
                'DD% on Max DD':[inc_DD],
                'NetP/MaxDD': [netpmaxdd],
                'Ret on Acc.t' : [returnOnAccount],
                'Max Trd Loss': [max_trade_loss],
                'Longest DD Day': [longest_drawdown],
                'Last Trd Day':[ultima_data],
                'Equity': [data_df['Profit/Loss'][0]]
                }
            file_name = report_name[:-4]
            lista = pd.concat([pd.DataFrame(data, index=[file_name]), lista])
            lista = lista.sort_values(['Net Profit'], ascending=[False])

        ### GRAPH >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        if not lista.empty:
            st.dataframe(lista, column_config={"Equity": st.column_config.LineChartColumn("Equity"), 
                                                "% Profit": st.column_config.NumberColumn(format='%.0f %%'),
                                                "DD% on Max DD": st.column_config.NumberColumn(format='%.0f %%'),
                                                "2 Mon Prof": st.column_config.CheckboxColumn(help="2 consecutive positive months"),
                                                "New Highs": st.column_config.CheckboxColumn(help="New highs reached in the last month"),
                                                }, use_container_width=True )
            col1, col2 = st.columns(2)
            with col1:
                uno, due = st.columns(2)  
                with uno:
                    optionX = st.selectbox('Select the metric for X',
                        ('Net Profit', 'Avg Trade', 'Num Trade', 'Profit Factor', 'Ret on Acc.t', '% Profit', 'NetP/MaxDD', 'Max Drawdown',
                        'DD% on Max DD', 'Max Trd Loss', 'Longest DD Day'))
                with due:
                    optionY = st.selectbox('Select the metric for Y',
                            ('Net Profit', 'Avg Trade', 'Num Trade', 'Profit Factor', 'Ret on Acc.t', '% Profit', 'NetP/MaxDD', 'Max Drawdown',
                            'DD% on Max DD', 'Max Trd Loss', 'Longest DD Day'), index=7)
                st.plotly_chart(fig2_scatter(lista, optionX, optionY), use_container_width=True)
            with col2:    
                uno, due = st.columns(2)    
                with uno:
                    option1 = st.selectbox('Select the metric for 1',
                        ('Net Profit', 'Avg Trade', 'Num Trade', 'Profit Factor', 'Ret on Acc.t', '% Profit', 'NetP/MaxDD', 'Max Drawdown',
                        'DD% on Max DD', 'Max Trd Loss', 'Longest DD Day'))
                with due:
                    option2 = st.selectbox('Select the metric for 2',
                        ('Net Profit', 'Avg Trade', 'Num Trade', 'Profit Factor', 'Ret on Acc.t', '% Profit', 'NetP/MaxDD', 'Max Drawdown',
                        'DD% on Max DD', 'Max Trd Loss', 'Longest DD Day'), index=7)  
                st.plotly_chart(fig3_bar(lista, option1, option2), use_container_width=True)
            
            option = st.selectbox('Select the metric', 
                    ('Net Profit', 'Avg Trade', 'Num Trade', 'Profit Factor', 'Ret on Acc.t', '% Profit', 
                    'NetP/MaxDD', 'Max Drawdown','DD% on Max DD', 'Max Trd Loss', 'Longest DD Day'))
            st.plotly_chart(compare_one_metric_reports(lista, option), use_container_width=True)
            ### GRAPH <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<