import streamlit as st
from streamlit_option_menu import option_menu
from function import *
from graphs import *
from MyDB import MyDB

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

def app():
    if 'isExpandedSP' not in st.session_state:
        st.session_state['isExpandedSP'] = True
    
    myDB = MyDB('armandinodinodello')

    symbol_df = myDB.get_symbol_list()
    symbol_map = symbol_df.set_index('Symbol Name')['Markets'].to_dict()
    symbol_margin_map = symbol_df.set_index('Symbol Name')['Overnight Margin'].to_dict()
    
    with st.expander("StrategyPro Dashboard", expanded=st.session_state['isExpandedSP']):
        if 'df_report_not_selected_strategyPro' not in st.session_state:
            tag_file_content = myDB.get_tag_file()
            reports_names = [entry['file_name'] for entry in tag_file_content]
            reports_tags = [" ".join(entry['tags']) for entry in tag_file_content]
            st.session_state['df_report_not_selected_strategyPro'] = pd.DataFrame({"Report Names": reports_names, "Tags": reports_tags})

        if 'df_report_selected_strategyPro' not in st.session_state:
            st.session_state['df_report_selected_strategyPro'] = pd.DataFrame({"Report Names": [], "Tags": []})

        # Search Bar & Radio Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        col1, col2 = st.columns(2)
        with col1:
            cerca = st.multiselect('Select the reports to analyze', st.session_state['df_report_not_selected_strategyPro'])
            query = st.text_input("Select Report to Compare")
            genre = st.radio( "FIlter bye", ["Name", "Tag"], index=0)    
            # Listener Search Bar ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            # & All Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if query:
                filter = 'Report Names' if (genre == 'Name') else 'Tags'
                df_filtered = st.session_state['df_report_not_selected_strategyPro'][st.session_state['df_report_not_selected_strategyPro'][filter].str.contains(query, case=False)]
                st.dataframe(df_filtered)
            elif cerca:
                df_filtered = st.session_state['df_report_not_selected_strategyPro'][st.session_state['df_report_not_selected_strategyPro']['Report Names'].isin(cerca)]
                st.dataframe(df_filtered)
            else:
                df_filtered = st.session_state['df_report_not_selected_strategyPro']
                st.dataframe(st.session_state['df_report_not_selected_strategyPro'])

            # Select Report Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if st.button("Select", use_container_width=True):
                st.session_state['df_report_selected_strategyPro'] = pd.concat([st.session_state['df_report_selected_strategyPro'], df_filtered], axis=0)
                st.session_state['df_report_selected_strategyPro']['Selected'] = True
                st.session_state['df_report_selected_strategyPro']['Weights'] = 1.0
                st.session_state['df_report_not_selected_strategyPro'] = pd.merge(st.session_state['df_report_not_selected_strategyPro'], df_filtered, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                st.session_state['isExpandedSP'] = True
                st.rerun()
            # Selected Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            with col2:
                
                initial_capital = st.number_input('Starting Capital', min_value=5000, max_value=10000000, value=100000, step=1000)
                
                st.write("Selected reports:")
                df = st.data_editor(st.session_state['df_report_selected_strategyPro'], disabled=("", "Report Names", "Tags"),
                                    column_config={"Weights": st.column_config.NumberColumn(),})  
                if not st.session_state['df_report_selected_strategyPro'].empty : 
                    weights_dict = df.set_index('Report Names')['Weights'].to_dict()            
                
                pul1, pul2 = st.columns(2)
                with pul1:
                    if st.button('Empty', use_container_width=True):
                        del st.session_state['df_report_not_selected_strategyPro']
                        del st.session_state['df_report_selected_strategyPro']
                        if 'df_report_to_strategyPro' in st.session_state:
                            del st.session_state['df_report_to_strategyPro']
                        st.rerun()
                with pul2:
                    if st.button('Save Portfolio', use_container_width=True):
                        if not st.session_state['df_report_selected_strategyPro'].empty and st.session_state['isGuest'] == False:
                            st.session_state['df_report_to_strategyPro'] = df[df['Selected'] == True]['Report Names'].to_numpy()
                            filtered_files = st.session_state['df_report_to_strategyPro']
                            # SETTING DATAFRAME DATAFRAME ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                            df_list = [prepare_df_strategy_pro(myDB.get_df_report(report_name), report_name) for report_name in st.session_state['df_report_to_strategyPro']]
                            
                            save_df = pd.concat(df_list)     
                            save_df['Max Contracts'] = save_df['StrategyName'].map(weights_dict) 

                            str_valori = np.array2string(filtered_files, separator=' ')
                            str_valori = str_valori.replace("[", "").replace("]", "").replace("'", "").replace("\n", "")
                            str_valori = "_".join(str_valori.split()) 
                            
                            save_df.columns = save_df.iloc[0]
                            save_df = save_df[1:]
                            
                            myDB.insert_portfolio(save_df, "Port-"+ str_valori)
                            st.success(f"Portfolio saved successfully: Port-"+ str_valori)
                        else:
                            st.error("Warning! This feature is for subscriber customers only.")

                if st.button('Analyze', use_container_width=True):
                    if not st.session_state['df_report_selected_strategyPro'].empty:
                        st.session_state['df_report_to_strategyPro'] = df[df['Selected'] == True]['Report Names'].to_numpy()
                        st.session_state['isExpandedSP'] = False
                        st.rerun()

    if 'df_report_to_strategyPro' in st.session_state:
        # SETTING DATAFRAME ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        df_list = [prepare_df_strategy_pro(myDB.get_df_report(report_name), report_name) for report_name in st.session_state['df_report_to_strategyPro']]
        tot_df = pd.concat(df_list) 
            
        tot_df['Max Contracts'] = tot_df['StrategyName'].map(weights_dict) 
        tot_df['ExitDate-Time'] = pd.to_datetime(tot_df['ExitDate-Time'], format="mixed", dayfirst=True) 
        tot_df['EntryDate-Time'] = pd.to_datetime(tot_df['EntryDate-Time'], format="mixed", dayfirst=True) 
        tot_df['TimeInTrading'] = (tot_df['ExitDate-Time'] - tot_df['EntryDate-Time']).astype(str) 
        tot_df = tot_df.set_index('ExitDate-Time')
        tot_df = tot_df.sort_values(by='ExitDate-Time')

        tot_df['Max Contracts'] = pd.to_numeric(tot_df['Max Contracts'], errors='coerce') 
        tot_df['Profit/Loss'] *= tot_df['Max Contracts'] 
        tot_df['MaxPositionProfit'] *= tot_df['Max Contracts'] 
        tot_df['MaxPositionLoss'] *= tot_df['Max Contracts'] 

        # WITHOUT INITIAL CAPITAL
        tot_df['Cumulative P/L'] = tot_df['Profit/Loss'].cumsum().astype('float32') 
        # WITH INITIAL CAPITAL
        #tot_df['Cumulative P/L'] = initial_capital + tot_df['Profit/Loss'].cumsum()

        tot_df['Max Equity'] = tot_df['Cumulative P/L'].cummax() 
        tot_df['Drawdown']  = tot_df['Max Equity'] - tot_df['Cumulative P/L']

        # WITHOUT INITIAL CAPITAL
        #tot_df['Drawdown %']  = round((tot_df['Max Equity'] - tot_df['Cumulative P/L']) / tot_df['Max Equity'] * 100,2) 
        # WITH INITIAL CAPITAL AND CALC EQUITY 
        #tot_df['Drawdown %'] = round((tot_df['Max Equity'] - tot_df['Cumulative P/L']) / initial_capital * 100, 2) 
        # WITH INITIAL CAPITAL AND CALC EQUITY 
        tot_df['Drawdown %'] = round(tot_df['Drawdown'] / initial_capital * 100, 2) 

        tot_df['Overnight Margin'] = tot_df['SymbolName'].map(symbol_margin_map)
        tot_df['Overnight Margin'] *= tot_df['Max Contracts']
        tot_df['Overnight Margin'].fillna(0)

        tot_df['Markets'] = tot_df['SymbolName'].map(symbol_map)
        tot_df['Markets'].fillna('miss')

        tot_df['Max Contracts'] = tot_df['Max Contracts'].astype('float32')
        tot_df['MaxPositionLoss'] = tot_df['MaxPositionLoss'].astype('float32')
        tot_df['MaxPositionProfit'] = tot_df['MaxPositionProfit'].astype('float32')
        tot_df['Profit/Loss'] = tot_df['Profit/Loss'].astype('float32')

        # STRATEGY DF
        str_df = divide_tot_df(tot_df, 'StrategyName')
        str_dfEquity = total_equity(str_df)
        str_df['Profit/Loss'] = str_df.sum(axis=1)
 
        # SYMBOL DF
        sym_df = divide_tot_df(tot_df, 'SymbolName')
        sym_dfEquity = total_equity(sym_df)
        sym_df['Profit/Loss'] = sym_df.sum(axis=1)
        
        # MARKETS DF
        mar_df = divide_tot_df(tot_df, 'Markets')
        mar_dfEquity = total_equity(mar_df)
        mar_df['Profit/Loss'] = mar_df.sum(axis=1)

        # POSITION DF
        pos_df = divide_tot_df(tot_df, 'MarketPosition')
        equity_LongShort = total_equity(pos_df)
        pos_df['Profit/Loss'] = pos_df.sum(axis=1)

        # NUM TRADE
        tot_numTrade = num_Trades(tot_df) 
        tot_winTrade = calc_winTrades(tot_df)
        tot_losTrade = calc_losTrades(tot_df)

        # TOTAL METRICS 
        tot_maxDrawdown = calc_maxDrawdown(tot_df['Drawdown'])                                              # MAX DRAWDOWN
        lastDrawdown = calc_lastDD(tot_df['Drawdown'])                                                      # LAST DD PERIODS
        lastDrawdownPerc = calc_inc(lastDrawdown, tot_maxDrawdown)                                          # LAST DD
        tot_netProfit = calc_netProfit(tot_df, 'Profit/Loss')                                               # NET PROFIT
        
        #tot_maxDrawdownPerc = calc_maxDDPerc(tot_maxDrawdown, tot_netProfit)                               # INCIDENCE DD % ON TOT DOF
        tot_maxDrawdownPerc = tot_df['Drawdown %'].max()                                                    # DD %

        tot_percProfit = calc_percProf(tot_winTrade, tot_numTrade)                                          # PERCENTAGE PROFIT
        tot_Avg = calc_AvgTrade(tot_netProfit, tot_numTrade)                                                # AVERAGE TRADE
        tot_GrossWin = calc_grossWin(tot_df, 'Profit/Loss')                                                 # GROSS WIN
        tot_GrossLos = calc_grossLoss(tot_df, 'Profit/Loss')                                                # GROSS LOSS
        tot_ProfitcFactor = calc_profictFactor(tot_GrossWin, tot_GrossLos)                                  # PROFICT FACTOR
        tot_averageWin = calc_averageWin(tot_GrossWin, tot_winTrade)                                        # AVERAGE WIN 
        tot_averageLoss = calc_averageLoss(tot_GrossLos, tot_losTrade)                                      # AVERAGE LOSS
        tot_returnOnAccount = cacl_retOnAcc(tot_maxDrawdown, tot_netProfit)                                 # RETURN ON ACCOUNT
        tot_avgDuration = duration_Trade(tot_df)                                                            # AVERAGE DURATION TRADE
        tot_maxTradeLoss = calc_maxTradeLoss(tot_df)                                                        # MAX TRADE LOSS
        tot_drawdown_periods, tot_longest_drawdown = calc_drawdown_periods(tot_df)                          # FIND BEST DRAWDOWN PERIODS



        menu = option_menu(
                menu_title=None,  
                options=["Charts", "Metrics", "Annual Metrics", "Trade Analysis", "Correlations", "Distribution", "Allocation", "Returns", 
                "Trade List", "Equity Control", "MonteCarlo", "What If?", "Volatility"], 
                icons=[" ", " ", " "," ", " ", " ", " ", " ", " ", " ", " ", " ", " "], 
                menu_icon="cast",  
                default_index=0,  
                orientation="horizontal",
            )

        ### CHARTS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Charts": 
            fig_portfolio = gen_equityDrawdown(tot_df, tot_df['Cumulative P/L'], tot_df['Drawdown'], tot_df['Drawdown %'])
            fig_portfolio_system = portplussingol(str_dfEquity, tot_df, tot_df['Drawdown %'])
            fig_portfolio_ddperiod = plot_drawdown_periods(tot_df, tot_drawdown_periods)
            fig_long_short = create_equity_line_plot(equity_LongShort, "", "Equity")

            if not mar_df.empty:
                fig_markets = create_equity_line_plot(mar_dfEquity, "", "Equity")
            
            fig_symbol = create_equity_line_plot(sym_dfEquity, "", "Equity")
            fig_system = create_equity_line_plot(str_dfEquity, "", "Equity") 
             
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(['Portfolio', 'Portfolio & System','Portfolio & DD Periods', 'Long/Short', 'Markets', 'Symbol', 'System'])
            with tab1:  
                st.plotly_chart(fig_portfolio, use_container_width=True)
            with tab2:
                st.plotly_chart(fig_portfolio_system, use_container_width=True)
            with tab3:
                st.plotly_chart(fig_portfolio_ddperiod, use_container_width=True)
            with tab4:
                st.plotly_chart(fig_long_short, use_container_width=True)
            with tab5: 
                if not mar_df.empty:
                    st.plotly_chart(fig_markets, use_container_width=True)
                else:
                    st.warning("No Markets insert in Symbol Manager")
            with tab6:
                st.plotly_chart(fig_symbol, use_container_width=True)
            with tab7:
                st.plotly_chart(fig_system, use_container_width=True)


        ### METRICS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Metrics":
            # DF 5%
            per_numTrade = int(tot_numTrade * 5 / 100) 
            per_df = tot_df[-per_numTrade:]
            per_winTrade = calc_winTrades(per_df) 
            per_losTrade = calc_losTrades(per_df)

            # DF 2 MONTHS
            current_month, previous_month, previous_year, two_months_ago_month, two_months_ago_year = creation_Month() 
            mon_df = twoMon_df(tot_df, two_months_ago_month, two_months_ago_year, previous_month, previous_year, current_month) 
            mon_numTrade = num_Trades(mon_df) 
            mon_winTrade = calc_winTrades(mon_df) 
            mon_losTrade = calc_losTrades(mon_df)

            ## CALC METRICS
            per_maxDrawdown = calc_maxDrawdown(per_df['Drawdown'])
            mon_maxDrawdown = calc_maxDrawdown(mon_df['Drawdown'])

            per_netProfit = calc_netProfit(per_df, 'Profit/Loss')
            mon_netProfit = calc_netProfit(mon_df, 'Profit/Loss')
          
            per_maxDrawdownPerc = per_df['Drawdown %'].max() 
            mon_maxDrawdownPerc = mon_df['Drawdown %'].max() 
       
            per_percProfit = calc_percProf(per_winTrade, per_numTrade)
            mon_percProfit = calc_percProf(mon_winTrade, mon_numTrade)
      
            per_Avg = calc_AvgTrade(per_netProfit, per_numTrade)
            mon_Avg = calc_AvgTrade(mon_netProfit, mon_numTrade)

            per_GrossWin = calc_grossWin(per_df, 'Profit/Loss') 
            mon_GrossWin = calc_grossWin(mon_df, 'Profit/Loss')
            
            per_GrossLos = calc_grossLoss(per_df, 'Profit/Loss')
            mon_GrossLos = calc_grossLoss(mon_df, 'Profit/Loss') 
           
            per_ProfitcFactor = calc_profictFactor(per_GrossWin, per_GrossLos)
            mon_ProfitcFactor = calc_profictFactor(mon_GrossWin, mon_GrossLos)
         
            per_averageWin = calc_averageWin(per_GrossWin, per_winTrade)
            mon_averageWin = calc_averageWin(mon_GrossWin, mon_winTrade)
          
            per_averageLoss = calc_averageLoss(per_GrossLos, per_losTrade)
            mon_averageLoss = calc_averageLoss(mon_GrossLos, mon_losTrade)
            
            per_returnOnAccount = cacl_retOnAcc(per_maxDrawdown, per_netProfit)
            mon_returnOnAccount = cacl_retOnAcc(mon_maxDrawdown, mon_netProfit)
            
            per_avgDuration = duration_Trade(per_df)
            mon_avgDuration = duration_Trade(mon_df)
            
            per_maxTradeLoss = calc_maxTradeLoss(per_df) 
            mon_maxTradeLoss = calc_maxTradeLoss(mon_df) 
            
            avg_set = int(tot_netProfit / len(str_df.resample('7D')))
            avg_men = int(tot_netProfit / len(str_df.resample('ME')))
            avg_ann = int(tot_netProfit / len(str_df.resample('YE')))
            
            newHigh = check_new_high(tot_df['Cumulative P/L'], previous_month, previous_year, current_month)
            
            twoMonthPos = check_last_two_months_positive(tot_df) 
            
            per_drawdown_periods, per_longest_drawdown = calc_drawdown_periods(per_df) 
            mon_drawdown_periods, mon_longest_drawdown = calc_drawdown_periods(mon_df) 

            col0, col1, col2, col3, col4, col5  = st.columns((1, 1, 1, 1, 1, 1))
            with col0:
                st.write("")
            with col1: 
                st.subheader("Tot Metrics")
                metriche_general(st, tot_netProfit, tot_GrossWin, tot_GrossLos, tot_ProfitcFactor, tot_Avg, tot_percProfit, 
                                tot_returnOnAccount, tot_averageWin, tot_averageLoss, tot_numTrade, tot_winTrade,
                                tot_losTrade, tot_maxDrawdown, tot_maxDrawdownPerc, tot_avgDuration, tot_maxTradeLoss,
                                tot_longest_drawdown)
            with col2: 
                st.subheader("Last 5%")
                metriche_general(st, per_netProfit, per_GrossWin, per_GrossLos, per_ProfitcFactor, per_Avg, per_percProfit,
                                per_returnOnAccount, per_averageWin, per_averageLoss, per_numTrade,
                                per_winTrade, per_losTrade, per_maxDrawdown, per_maxDrawdownPerc, per_avgDuration, per_maxTradeLoss,
                                per_longest_drawdown)
            with col3: 
                st.subheader("Last 2 Months")
                metriche_general(st, mon_netProfit, mon_GrossWin, mon_GrossLos, mon_ProfitcFactor, mon_Avg,
                                mon_percProfit, mon_returnOnAccount, mon_averageWin,
                                mon_averageLoss, mon_numTrade, mon_winTrade, mon_losTrade,
                                mon_maxDrawdown, mon_maxDrawdownPerc, mon_avgDuration, mon_maxTradeLoss,
                                mon_longest_drawdown)
            with col4: 
                metriche_finali(st, avg_set, avg_men, avg_ann, twoMonthPos, newHigh,
                                lastDrawdown, lastDrawdownPerc)
            with col5: 
                st.write("")
            
        ### ANNUAL METRICS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  
        if menu == "Annual Metrics":          
            colcopy = ['Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss', 'TimeInTrading', 'Drawdown', 'Cumulative P/L']
            years_df = tot_df[colcopy].copy()
            anni = years_df.index.year 
            lista = pd.DataFrame()
            
            for anno in set(anni):
                df = years_df[anni == anno].copy()
                max_drawdown = calc_maxDrawdown(df['Drawdown'])   
                df['Drawdown %'] = round(df['Drawdown'] / initial_capital * 100, 2) 
                maxDrawdownPerc = df['Drawdown %'].max() 

                numTrades = num_Trades(df)     
                netProfit = calc_netProfit(df, 'Profit/Loss') 
                GrossWin = calc_grossWin(df, 'Profit/Loss')
                GrossLos = calc_grossLoss(df, 'Profit/Loss')
                netpmaxdd = calc_NetpOnMaxdd(netProfit, max_drawdown) 
                avg = int(calc_AvgTrade(netProfit, numTrades))
                ProfitcFactor = calc_profictFactor(GrossWin, GrossLos) 
                win_trades = calc_winTrades(df)
                percProfit =int(calc_percProf(win_trades, numTrades))         
                returnOnAccount = cacl_retOnAcc(max_drawdown, netProfit)
                avg_Duration = duration_Trade(df)
                max_trade_loss = calc_maxTradeLoss(df)  

                new_dataframe = pd.DataFrame({'Profit/Loss': df['Cumulative P/L']})
                data_df = pd.DataFrame({"Profit/Loss": [new_dataframe['Profit/Loss'].tolist()]}) 
                yr_drawdown_periods, yr_longest_drawdown = calc_drawdown_periods(df)  

                data = {
                    'Net Profit': [netProfit],
                    'Avg Trade': [avg],
                    'Num Trade':[numTrades],
                    'Profict Factor':[ProfitcFactor],
                    'Ret on Acc.t' : [returnOnAccount],
                    '% Profit': [percProfit],
                    'NetP/MaxDD': [netpmaxdd],
                    'Max Drawdown': [max_drawdown],
                    'Max Drawdown %': [maxDrawdownPerc],
                    'Max Trade Loss': [max_trade_loss],
                    'Avg Duration': [avg_Duration],
                    'Days in DD': [yr_longest_drawdown],
                    'Equity': [data_df['Profit/Loss'][0]],
                    }
                
                file_name = str(anno)
                data['Date'] = pd.to_datetime(file_name, format='%Y')
                lista = pd.concat([pd.DataFrame(data, index=[file_name]), lista])

            if not lista.empty:
                lista = lista.sort_values('Date', ascending=False)
                lista = lista.drop('Date', axis=1)
                st.dataframe(lista, column_config={
                    "Equity": st.column_config.LineChartColumn("Equity"), 
                    "% Profit": st.column_config.NumberColumn(format='%.0f %%'),}, use_container_width=True )
             
                option = st.selectbox('Select the metric',
                        ('Net Profit', 'Avg Trade', 'Num Trade', 'Profict Factor', 'Ret on Acc.t', '% Profit', 'NetP/MaxDD', 'Max Drawdown', 'Max Drawdown %', 'Max Trade Loss',
                       'Days in DD'))
                fig5 = px.bar(lista, y=option, text=option)
                fig5.update_traces( hovertemplate=None,  textposition='outside')
                fig5.update_layout(title =f"{option} in Years", xaxis_title="Year")
                st.plotly_chart(fig5, use_container_width=True)
               
                
        ### TRADE ANALISYS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Trade Analysis": 
            colcopy = ['Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss', 'TimeInTrading', 'EntryDate-Time']
            df = tot_df[colcopy].copy()
            df['Color'] = ['red' if x < 0 else 'blue' for x in df['Profit/Loss']]
            df['TradeResult'] = df['Profit/Loss'].apply(lambda x: 'Profit' if x > 0 else 'Loss')
            df['MAE'] = df[['Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss']].min(axis=1)
            df['MFE'] = df[['Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss']].max(axis=1)
            
            serie_Consecutiva = processDf_consecutive(df)
            positive = positive_sequence(serie_Consecutiva)
            negative = negative_sequence(serie_Consecutiva)
            serie = pd.concat([positive, negative], axis=1)
            
            fig = tradeTime_graph(df)
            df['MaxPositionProfit'] = df['MaxPositionProfit'].apply(lambda x: max(x, 0))
            fig_mae = mae_mfe(df, 'MAE', 'Maximum Adverse Excursion (MAE) Table')
            fig_mfe = mae_mfe(df, 'MFE', 'Maximum Favorable Excursion (MFE) Table')
            fig_tradeinTime = tradeinTime(df)

            st.plotly_chart(fig, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(fig_mae, use_container_width=True)
            with col2: st.plotly_chart(fig_mfe, use_container_width=True)
            col3, col4 = st.columns((1,4))
            with col3: st.dataframe(serie, use_container_width=True)
            with col4: st.plotly_chart(fig_tradeinTime, use_container_width=True)
            

        ### CORRELATIONS +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Correlations":
            categories_period = ["Day", "Week", "Month", "3 Month", "1 Year", "2 Year"]
            caterogies_corr = ["Strategy", "Symbol", "Market", "Long/Short"]

            col1, col2, col3 = st.columns(3)            
            with col1: selected_categories = st.selectbox("Correlation by:", caterogies_corr, index=0)
            with col2: selected_period = st.selectbox("Correlation Period:", categories_period, index=4)
            with col3: choice = st.radio("", ['Profit & Loss', 'Only Profit', 'Only Loss'])

            if selected_period == "Day": period = 'D' 
            elif selected_period == "Week": period = '7D' 
            elif selected_period == "Month": period = '30D'
            elif selected_period == "3 Month": period = '90D'
            elif selected_period == "1 Year": period = '365D'
            else: period = '730D' 

            if selected_categories == "Strategy": df = create_corr_df(str_df)
            elif selected_categories == "Symbol": df = create_corr_df(sym_df)
            elif selected_categories == "Market": df = create_corr_df(mar_df)
            else: df = create_corr_df(pos_df)

            if choice == 'Only Profit':
                df = replace_values(df, True)
            if choice == 'Only Loss':
                df = replace_values(df, False)
                        
            roll_corr, df_avg_corr = rolling_correlation(df, list(combinations(df.columns, 2)), period)
  
            corrStyler = apply_corr_style(df_avg_corr)
            st.dataframe(corrStyler, use_container_width=True)

            if not roll_corr.empty:
                fig = create_equity_line_plot(roll_corr, selected_categories + " Rolling Correlation", "Correlation")
                st.plotly_chart(fig, use_container_width=True)


        ### DISTRIBUTION  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Distribution":
            colcopy = ['Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss']
            df = tot_df[colcopy].copy()

            mu, sigma = calc_normal_distribution(df, 'Profit/Loss') 
            
            count_sett = ret_sett(df, 'count')
            count_gmo = ret_dayMonth(df, 'count')
            count_hou = ret_hour(df, 'count')

            if -1 in pos_df.columns and 1 in pos_df.columns:
                pos_dfLong = pos_df[(pos_df[1] != 0)]
                pos_dfShort = pos_df[(pos_df[-1] != 0)]

            tabProfLoss, tabTrade = st.tabs(['Profit/Loss', 'Trade Distribution'])
            with tabProfLoss:
                plot_normal_distribution(mu, sigma, df, 'Profit/Loss')
                plot_distr_doppia(df, 'MaxPositionProfit', 'green', df, 'MaxPositionLoss', 'red')
                if -1 in pos_df.columns and 1 in pos_df.columns:
                    plot_distr_doppia(pos_dfLong, 1, 'blue', pos_dfShort, -1, 'purple')

            with tabTrade:
                col1, col2 = st.columns(2)
                with col1: grafico_settimanale(count_sett, "Trade Count") 
                with col2: grafico_ggMese(count_gmo, 'Sum of Trade for day of the month')
                grafico_ritorni_orari(count_hou, 'Sum of Trade for each hour of the day')


        ### ALLOCATION +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Allocation":
            # ASSET ALLOCATION 
            colcopy = ['EntryDate-Time', 'Profit/Loss', 'StrategyName', 'SymbolName', 'MarketPosition', 'Markets', 'Overnight Margin']
            df = tot_df[colcopy].copy()

            sumColName = st.selectbox('Select the comparison criterion',    
                                        ('Profit/Loss', 'Overnight Margin'))

            allocation_NetProfit = calc_assAll(df, 'StrategyName', sumColName)
            allocation_Symbol = calc_assAll(df, 'SymbolName', sumColName)
            allocation_Position = calc_assAll(df, 'MarketPosition', sumColName)
            allocation_Market = calc_assAll(df,'Markets', sumColName)

            df.reset_index(inplace=True) 
            start_date = df.loc[0, 'EntryDate-Time'].date()
            end_date = df['ExitDate-Time'].max().date()
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_df = pd.DataFrame({'Date': date_range})

            active_df = pd.DataFrame(index=date_df['Date'])
            active_df['ActiveStrategies'] = 0

            for _, row in df.iterrows():
                start_date = row['EntryDate-Time'].normalize()
                end_date = row['ExitDate-Time'].normalize()
                strategy_name = row['StrategyName']
                active_df.loc[start_date:end_date, strategy_name] = 1

            active_df.fillna(0, inplace=True)
            active_df['ActiveStrategies'] = active_df.sum(axis=1)
            active_df = active_df.astype('int16')
            
            margin_df = pd.DataFrame(index=date_df['Date'])
            
            for _, row in df.iterrows():
                start_date = row['EntryDate-Time'].normalize()
                end_date = row['ExitDate-Time'].normalize()
                margin_value = row['Overnight Margin']
                strategy_name = row['StrategyName']
                margin_df.loc[start_date:end_date, strategy_name] = margin_value

            margin_df.fillna(0, inplace=True)
            margin_df['Total Margin'] = margin_df.sum(axis=1)
            margin_df = margin_df.astype('int32')
        
            strategies_counts = active_df.groupby('ActiveStrategies').size()
            strategies_counts_df = strategies_counts.reset_index()
            strategies_counts_df.columns = ['Active strategies', 'Active days']
            strategies_counts_df = strategies_counts_df.set_index('Active strategies')

            statistics_active_strategies = active_df['ActiveStrategies'].describe()
            mean_active_strategies = statistics_active_strategies['mean']
            median_active_strategies = statistics_active_strategies['50%']
            
            statistics_active_margins = margin_df['Total Margin'].describe()
            mean_margin = statistics_active_margins['mean']
            median_margin = statistics_active_margins['50%']
            min_margin = statistics_active_margins['min']
            max_margin =statistics_active_margins['max']

            fig = create_overTime(active_df, 'ActiveStrategies', ' ', 'Time', 'Number of Strategies') 
            fig2 = create_overTime(margin_df, 'Total Margin', ' ', 'Time', 'Margin') 

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                plot_assAll(allocation_NetProfit['StrategyName'], allocation_NetProfit[sumColName], 'Strategy Name')
            with col2:
                plot_assAll(allocation_Symbol['SymbolName'], allocation_Symbol[sumColName], 'Symbol Name')
            with col3:
                plot_assAll(allocation_Position['MarketPosition'], allocation_Position[sumColName], 'Market Position')
            with col4:
                plot_assAll(allocation_Market['Markets'], allocation_Market[sumColName], 'Market')
            
            st.subheader("Active Strategies")
            colStr1, colStr2 = st.columns((0.5,3))
            with colStr1:
                st.write("Average active strategies daily:", round(mean_active_strategies, 1))
                st.write("Median daily active strategies:", round(median_active_strategies, 1))
                st.dataframe(strategies_counts_df, use_container_width=True)
            with colStr2:
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Margin over Time")
            colMar1, colMar2 = st.columns((0.5,3))
            with colMar1:
                st.write("Average daily margins:", round(mean_margin))
                st.write("Median daily margins:", round(median_margin))
                st.write("Min margins:", round(min_margin))
                st.write("Max margins:", round(max_margin))
            with colMar2:
                st.plotly_chart(fig2, use_container_width=True)


        ### RETURNS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Returns":
            colcopy = ['Profit/Loss']
            df = tot_df[colcopy].copy()
            
            ret_set = ret_sett(df, 'sum') 
            ret_men = ret_mens(df)
            ret_gmo = ret_dayMonth(df, 'sum') 
            ret_ann = ret_year(df)
            ret_hou = ret_hour(df, 'sum')
            
            if 1 in pos_df.columns and -1 in pos_df.columns:
                ret_setLS = ret_sett(pos_df, 'sum', 1, -1) 
                ret_menLS = ret_mens(pos_df)
                ret_gmoLS = ret_dayMonth(pos_df, 'sum', 1, -1) 
                ret_annLS = ret_year(pos_df, 1, -1)
                ret_houLS = ret_hour(pos_df, 'sum', 1, -1)
        
            tablongshort, tablongorshort = st.tabs(['Long&Short', 'Long vs. Short'])
            with tablongshort:
                rendimenti_mensili(df) 
                col9, col10  = st.columns(2, gap="small")
                with col9: grafico_annuali(ret_ann)  
                with col10: grafico_mensile(ret_men) 

                col11, col12 = st.columns(2, gap="small")
                with col11: grafico_settimanale(ret_set, 'Sum of Profit/Loss by day of the week') 
                with col12: grafico_ggMese(ret_gmo, 'Sum of Profit/Loss for day of the month') 
                grafico_ritorni_orari(ret_hou, 'Sum of Profit/Loss for each hour of the day')
                
            with tablongorshort:
                if 1 in pos_df.columns and -1 in pos_df.columns:
                    col9, col10  = st.columns(2, gap="small")
                    with col9: grafico_annuali(ret_annLS, 1, -1) 
                    with col10: grafico_mensile(ret_menLS, 1, -1)

                    col11, col12 = st.columns(2, gap="small")
                    with col11: grafico_settimanale(ret_setLS, 'Sum of Long/Short by day of the week', 1, -1)
                    with col12: grafico_ggMese(ret_gmoLS,'Sum of Long/Short for each day of the month', 1,-1) 
                    grafico_ritorni_orari(ret_houLS, 'Sum of Long/Short for each hour of the day', 1, -1)
                else: 
                    st.write("Long/Short operations only")
                

        ### TRADE LIST +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Trade List":
            tot_df.reset_index(inplace=True) 
            tot_df = tot_df[['EntryDate-Time', 'ExitDate-Time', 'EntryPrice', 'ExitPrice', 'SymbolName', 'Markets', 'Overnight Margin', 'MarketPosition', 
            'Max Contracts', 'Profit/Loss', 'MaxPositionProfit', 'MaxPositionLoss', 'StrategyName', 'TimeInTrading', 'Cumulative P/L']]
            st.dataframe(tot_df, use_container_width = True)


        ### EQUITY CONTROL +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Equity Control":
            
            colcopy = ['Profit/Loss', 'Cumulative P/L']
            df = tot_df[colcopy].copy()
            df.rename(columns={'Cumulative P/L': 'Original Equity'}, inplace=True)

            categories = ["Sma", "BB Low", "Mov Avg Add Contracts"]
            
            col1, col2, col3 = st.columns(3)
            with col1: selected_categories = st.selectbox("Control By:", categories)
            with col2: lenMa = st.number_input('SMA length', min_value=10, max_value=200, value=50, step=10)

            if selected_categories == "Sma":
                df['Control curve'] = df['Original Equity'].rolling(window=lenMa).mean() 
                df['Control'] = df.apply(lambda row: 0 if pd.isna(row['Control curve']) or row['Original Equity'] > row['Control curve'] else 1, axis=1)
                df['Control'] = df['Control'].shift(fill_value=1)
                df['Profit/Loss Contr'] = df.apply(lambda row: row['Profit/Loss'] if row['Control'] == 0 else 0, axis=1) 
                df['Controlled equity'] = df['Profit/Loss Contr'].cumsum() 

            elif selected_categories == "BB Low":
                with col3: 
                    lennDev = st.number_input('DEV length', min_value=1, max_value=5, value=2, step=1)

                    df['SMA'] = df['Original Equity'].rolling(window=lenMa).mean()
                    df['STD'] = df['Original Equity'].rolling(window=lenMa).std() 
                    df['Control curve'] = df['SMA'] - lennDev * df['STD']
                    df = df.drop(['SMA', 'STD'], axis=1)
                    df['Control'] = df.apply(lambda row: 0 if pd.isna(row['Control curve']) or row['Original Equity'] > row['Control curve'] else 1, axis=1)
                    df['Control'] = df['Control'].shift(fill_value=1) 
                    df['Profit/Loss Contr'] = df.apply(lambda row: row['Profit/Loss'] if row['Control'] == 0 else 0, axis=1) 
                    df['Controlled equity'] = df['Profit/Loss Contr'].cumsum()

            else: 
                with col3: 

                    moltiplicatore = st.number_input('Num Contracts', min_value=1, max_value=10, value=2, step=1)
                    df['Control curve'] = df['Original Equity'].rolling(window=lenMa).mean() 
                    df['Control'] = df.apply(lambda row: 0 if pd.isna(row['Control curve']) or row['Original Equity'] > row['Control curve'] else 1, axis=1)
                    df['Control'] = df['Control'].shift(fill_value=1) 
                    def calculate_profit_loss_contr(row):
                        if row['Control'] == 0:
                            return row['Profit/Loss']
                        elif row['Control'] == 1:
                            return row['Profit/Loss'] * moltiplicatore
                        else:
                            return None 
                    df['Profit/Loss Contr'] = df.apply(calculate_profit_loss_contr, axis=1)
                    df['Controlled equity'] = df['Profit/Loss Contr'].cumsum()

            con_NetProfit = calc_netProfit(df, 'Profit/Loss Contr')
            con_GrossWin = calc_grossWin(df, 'Profit/Loss Contr')
            con_GrossLos = calc_grossLoss(df, 'Profit/Loss Contr')
            con_ProfitcFactor = calc_profictFactor(con_GrossWin, con_GrossLos)
            con_numTrade = df[df["Control"] == 0]["Profit/Loss Contr"].count()
            con_winTrade = df[(df["Control"] == 0) & (df["Profit/Loss Contr"] > 0)]["Profit/Loss Contr"].count()
            con_losTrade = df[(df["Control"] == 0) & (df["Profit/Loss Contr"] < 0)]["Profit/Loss Contr"].count()
            con_Avg = calc_AvgTrade(con_NetProfit, con_numTrade)
            con_percProfit = calc_percProf(con_winTrade, con_numTrade)

            df['Max Equity'] = df['Controlled equity'].cummax() 
            df['Drawdown']  = df['Max Equity'] - df['Controlled equity'] 
            con_maxDrawdown = calc_maxDrawdown(df['Drawdown'])

            df['Drawdown %'] = round(df['Drawdown'] / initial_capital * 100, 2)
            con_maxDrawdownPerc = df['Drawdown %'].max() 

            con_returnOnAccount = cacl_retOnAcc(con_maxDrawdown, con_NetProfit)
            con_averageWin = calc_averageWin(con_GrossWin, con_winTrade)
            con_averageLoss = calc_averageLoss(con_GrossLos, con_losTrade)
            con_maxTradeLoss = int(df['Profit/Loss Contr'].min()) 

            con_drawdown_periods, con_longest_drawdown = calc_drawdown_periods(df) 

            fig = plot_equity_control(df)
            colOri, colCon, colGra = st.columns((0.5, 0.5, 3))
            with colOri:
                st.subheader("Original Metrics")
                metriche_general(st, tot_netProfit, tot_GrossWin, tot_GrossLos, tot_ProfitcFactor, tot_Avg, tot_percProfit, 
                                tot_returnOnAccount, tot_averageWin, tot_averageLoss, tot_numTrade, tot_winTrade,
                                tot_losTrade, tot_maxDrawdown, tot_maxDrawdownPerc, tot_avgDuration, tot_maxTradeLoss, tot_longest_drawdown)
            with colCon:
                st.subheader("Control Metrics")
                metriche_general(st, con_NetProfit, con_GrossWin, con_GrossLos, con_ProfitcFactor, con_Avg, con_percProfit, 
                                con_returnOnAccount, con_averageWin, con_averageLoss, con_numTrade, con_winTrade,
                                con_losTrade, con_maxDrawdown, con_maxDrawdownPerc, tot_avgDuration, con_maxTradeLoss, con_longest_drawdown)

            with colGra: st.plotly_chart(fig, use_container_width=True)


        ### MONTECARLO ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "MonteCarlo":
            colcopy = ['Profit/Loss']
            df = tot_df[colcopy].copy()
            df.reset_index(drop=True, inplace=True)
            equity = df['Profit/Loss'].cumsum()

            col1, col2, col3 = st.columns(3)

            with col1: num_simulations = st.number_input('Number of Simulations', min_value=5, max_value=500, value=10, step=10)
            with col2: num_future_trades = st.number_input('Number of future Operations', min_value=10, max_value=500, value=100, step=20)
            with col3: fun_montecarlo = st.radio("Montecarlo Type",["Randomized Only", "Randomize Returns Variate"] )

            tab_Montecarlo, tab_Simulation = st.tabs(["Montecarlo", "Simulation"])

            with tab_Montecarlo: 
                cumulative_returns, df_profitloss = monte_carlo_randomized(df, num_simulations, fun_montecarlo)

                cumulative_returns_all = cumulative_returns.copy()
                cumulative_returns_all['Original'] = equity 
                cumulative_returns_all = cumulative_returns_all.astype(int)
                cumulative_returns_all.columns = [str(col) for col in cumulative_returns_all.columns]

                metrics_results = calculate_metrics_simulate(cumulative_returns, df_profitloss, initial_capital)
                sim_Max_Drawdown = int(metrics_results["MaxDrawdown"].mean())
                sim_Max_Drawdown_perc = int(metrics_results["Perc Drawdown"].mean())
                sim_net_Profit = int(metrics_results["Net Profit"].mean())
                sim_ret_acc = int(metrics_results["Return On Acc"].mean())

                distr_netProfit = distribution_Metrics(metrics_results, "Net Profit")
                distr_MaxDrawdown = distribution_Metrics(metrics_results, "MaxDrawdown")
                distr_retAcc = distribution_Metrics(metrics_results, "Return On Acc")
                distr_MaxDrawdown_perc = distribution_Metrics(metrics_results, "Perc Drawdown")
                
                fig_equityMonte = plot_Montecarlo(cumulative_returns_all)
                fig_distr_NetP = px.bar(distr_netProfit, x="Net Profit", y='Frequency')
                fig_distr_MaxDD = px.bar(distr_MaxDrawdown, x="MaxDrawdown", y='Frequency')
                fig_distr_retAcc = px.bar(distr_retAcc, x="Return On Acc", y='Frequency')
                fig_distr_MaxDD_perc = px.bar(distr_MaxDrawdown_perc, x="Perc Drawdown", y='Frequency')

                col1, col2 = st.columns((1,5.5))
                
                with col1:
                    st.subheader("Original")
                    tot_netProfit = locale.currency(tot_netProfit, grouping=True)
                    st.write(":blue[Net Profit:]", tot_netProfit)
                    tot_maxDrawdown = locale.currency(tot_maxDrawdown, grouping=True)
                    st.write(":red[Max Drawdown:]", tot_maxDrawdown)
                    st.write(":red[Max Drawdown %: ] {:.2f}%".format(tot_maxDrawdownPerc))
                    st.write(":green[Return on Account:] {:.0f}".format(tot_returnOnAccount)) 
                    st.subheader("Avg Simulate")
                    sim_net_Profit = locale.currency(sim_net_Profit, grouping=True)
                    st.write(":blue[AVG Net Profit:]", sim_net_Profit)
                    sim_Max_Drawdown = locale.currency(sim_Max_Drawdown, grouping=True)
                    st.write(":red[AVG Max Drawdown:]", sim_Max_Drawdown)
                    st.write(":red[Max Drawdown %: ] {:.2f}%".format(sim_Max_Drawdown_perc))
                    st.write(":green[AVG Return on Account:] {:.0f}".format(sim_ret_acc))
                
                with col2: st.plotly_chart(fig_equityMonte, use_container_width=True)

                st.subheader("Frequency")
                d1, d2, d3, d4 = st.columns(4)
                with d1: st.plotly_chart(fig_distr_NetP, use_container_width = True)
                with d2: st.plotly_chart(fig_distr_MaxDD, use_container_width = True)
                with d3: st.plotly_chart(fig_distr_retAcc, use_container_width = True)
                with d4: st.plotly_chart(fig_distr_MaxDD_perc, use_container_width = True)
            with tab_Simulation:                 
                historical_equity, simulated_equity, p5, p95 = calc_equity_simul(df, num_simulations, num_future_trades)
                plot_equity(historical_equity, simulated_equity, p5, p95, num_future_trades)


        ### WHAT IF ANALYSIS ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "What If?":         
            colcopy = ['Profit/Loss', 'TimeInTrading', 'MarketPosition', 'Drawdown']
            df = tot_df[colcopy].copy()
            df_sorted = df.sort_values(by='Profit/Loss', ascending=False)
            
            anni_da_escludere = []
            mesi_da_escludere = []
            operazione = []
            
            with st.form("ciao"):
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    anni_da_escludere = st.multiselect('YEARS to exclude:', sorted(df.index.year.unique()))
                with col2:
                    sel_mesi_da_escludere = st.multiselect('MONTHS to exclude:',
                        ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
                    mese_a_numero = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
                    mesi_da_escludere = [mese_a_numero[mese] for mese in sel_mesi_da_escludere]
                with col3:
                    sel_giorni_da_escludere = st.multiselect('WEEK DAY to exclude:', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                    giorno_a_numero = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
                    giorni_da_escludere = [giorno_a_numero[mese] for mese in sel_giorni_da_escludere]
                with col4:
                    ore_da_escludere = st.multiselect('Select the hours to exclude:',
                    ['0','1','2','3','4','5','6','7','8','9','10','11','12',
                    '13','14','15','16','17','18','19','20','21','22','23'])
                with col5:
                    operation = st.radio("Select the operations to exclude", ["None", "Long", "Short"])
                    if operation == "None": operazione = []
                    elif operation == "Long": operazione = [1]
                    else: operazione = [-1]
                with col6:
                    lenPerc = st.number_input('Eliminiating Best % Trades', min_value=0, max_value=95, value=0, step=5)
                    num_records_to_exclude = int(len(df_sorted) * (lenPerc/100))
                    df_excluded = df_sorted.iloc[num_records_to_exclude:]

                submitted = st.form_submit_button("Submit", use_container_width=True)

            if submitted:
                anni_da_escludere = [int(anno) for anno in anni_da_escludere]
                mesi_da_escludere = [int(mese) for mese in mesi_da_escludere]
                giorni_da_escludere = [int(giorno) for giorno in giorni_da_escludere]
                ore_da_escludere = [int(ora) for ora in ore_da_escludere]
                operazione = [int(op) for op in operazione]

            df = df[
                (~df.index.year.isin(anni_da_escludere)) &
                (~df.index.month.isin(mesi_da_escludere)) &
                (~df.index.dayofweek.isin(giorni_da_escludere)) &
                (~df.index.hour.isin(ore_da_escludere)) &
                (~df['MarketPosition'].isin(operazione)) &
                (df.index.isin(df_excluded.index))]

            df['Cumulative P/L'] = df['Profit/Loss'].cumsum()     
            tot_df_subset = tot_df[['Cumulative P/L']].rename(columns={'Cumulative P/L': 'Original'})
            df_subset = df[['Cumulative P/L']].rename(columns={'Cumulative P/L': 'Modified'})
            merged_df = pd.merge(tot_df_subset, df_subset, left_index=True, right_index=True, how='outer')
            merged_df.ffill(inplace=True)

            fig2 = create_equity_line_plot(merged_df, "", "Equity")
            
            con_NetProfit = calc_netProfit(df, 'Profit/Loss')
            con_GrossWin = calc_grossWin(df, 'Profit/Loss')
            con_GrossLos = calc_grossLoss(df, 'Profit/Loss')
            con_ProfitcFactor = calc_profictFactor(con_GrossWin, con_GrossLos)
            con_numTrade = df["Profit/Loss"].count()
            con_winTrade = df[(df["Profit/Loss"] > 0)]["Profit/Loss"].count()
            con_losTrade = df[(df["Profit/Loss"] < 0)]["Profit/Loss"].count()
            con_Avg = calc_AvgTrade(con_NetProfit, con_numTrade)
            con_percProfit = calc_percProf(con_winTrade, con_numTrade)
            
            df['Max Equity'] = df['Cumulative P/L'].cummax() 
            df['Drawdown']  = df['Max Equity'] - df['Cumulative P/L'] 
            con_maxDrawdown = calc_maxDrawdown(df['Drawdown']) 
            df['Drawdown %'] = round(df['Drawdown'] / initial_capital * 100, 2)
            con_maxDrawdownPerc = df['Drawdown %'].max() 
            
            con_returnOnAccount = cacl_retOnAcc(con_maxDrawdown, con_NetProfit)
            con_averageWin = calc_averageWin(con_GrossWin, con_winTrade)
            con_averageLoss = calc_averageLoss(con_GrossLos, con_losTrade)
            con_avg_Duration = duration_Trade(df)
            if not df.empty: con_maxTradeLoss = int(df['Profit/Loss'].min()) 
            else: con_maxTradeLoss = 0
            con_drawdown_periods, con_longest_drawdown = calc_drawdown_periods(df)  
            
            colOri, colCon, colGra = st.columns((0.5, 0.5, 3))
            with colOri:
                st.subheader("Original Metrics")
                metriche_general(st, tot_netProfit, tot_GrossWin, tot_GrossLos, tot_ProfitcFactor, tot_Avg, tot_percProfit, 
                                tot_returnOnAccount, tot_averageWin, tot_averageLoss, tot_numTrade, tot_winTrade,
                                tot_losTrade, tot_maxDrawdown, tot_maxDrawdownPerc, tot_avgDuration, tot_maxTradeLoss, tot_longest_drawdown)
            with colCon:
                st.subheader("Control Metrics")
                metriche_general(st, con_NetProfit, con_GrossWin, con_GrossLos, con_ProfitcFactor, con_Avg, con_percProfit, 
                                con_returnOnAccount, con_averageWin, con_averageLoss, con_numTrade, con_winTrade,
                                con_losTrade, con_maxDrawdown, con_maxDrawdownPerc, con_avg_Duration, con_maxTradeLoss, con_longest_drawdown)
            with colGra: st.plotly_chart(fig2, use_container_width=True)

        ### VOLATILITY ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if menu == "Volatility":   
            window_size = st.slider("Select the rolling period", min_value=5, max_value=200, value=5, step=5)
            
            colcopy = ['Profit/Loss']
            df = tot_df[colcopy].copy()
            df['Profit/Loss'] = pd.to_numeric(df['Profit/Loss'], errors='coerce')
            df['Profit/Loss'] = df['Profit/Loss'].ffill()  
            df['Daily_Return'] = df['Profit/Loss'].pct_change()
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df['Volatility'] = df['Daily_Return'].rolling(window_size).std().fillna(0) 
            
            std_mean = int(df['Volatility'].mean())
            std_std = int(df['Volatility'].std())
            std_min = int(df['Volatility'].min())
            std_max = int(df['Volatility'].max())
            std_quantile_25 = int(df['Volatility'].quantile(0.25))
            std_quantile_50 = int(df['Volatility'].quantile(0.50))
            std_quantile_75 = int(df['Volatility'].quantile(0.75))

            col1, col2 = st.columns((1,5.5))
            with col1:
                st.write("Mean:", std_mean)
                st.write("Std:", std_std)
                st.write("Min", std_min) 
                st.write("Max:", std_max) 
                st.write("Quantile 25%:", std_quantile_25) 
                st.write("Quantile 50%:", std_quantile_50) 
                st.write("Quantile 75%:", std_quantile_75) 

            with col2: 
                st.line_chart(df, y="Volatility")



