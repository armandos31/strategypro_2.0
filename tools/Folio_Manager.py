import streamlit as st
from MyDB import MyDB
from function import *
from graphs import *
from streamlit_option_menu import option_menu

def app():

    if 'isExpandedFM' not in st.session_state:
        st.session_state['isExpandedFM'] = True
        
    if st.session_state['isGuest']:
        user_id = ''
    else:
        user_id = st.session_state['username']
        
    myDB = MyDB(user_id, st.session_state['isGuest'])
    symbol_df = myDB.get_symbol_list()

    if user_id == 'armandinodinodello':
         with st.expander("Folio Dashboard", expanded=st.session_state['isExpandedFM']):
            if 'df_report_not_selected_folioManager' not in st.session_state:
                tag_file_content = myDB.get_tag_file()
                reports_names = [entry['file_name'] for entry in tag_file_content]
                reports_tags = [" ".join(entry['tags']) for entry in tag_file_content]
                st.session_state['df_report_not_selected_folioManager'] = pd.DataFrame({"Report Names":reports_names, "Tags":reports_tags})

            if 'df_report_selected_folioManager' not in st.session_state:
                st.session_state['df_report_selected_folioManager'] = pd.DataFrame({"Report Names": [], "Tags": []})

            # Search Bar & Radio Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            col1, col2 = st.columns(2)
            with col1:
                cerca = st.multiselect('Select the reports to analyze', st.session_state['df_report_not_selected_folioManager'])
                query = st.text_input("Select Report to Compare")
                genre = st.radio( "FIlter bye", ["Name", "Tag"], index=0)    
                # Listener Search Bar ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                # & All Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if query:
                    filter = 'Report Names' if (genre == 'Name') else 'Tags'
                    df_filtered = st.session_state['df_report_not_selected_folioManager'][st.session_state['df_report_not_selected_folioManager'][filter].str.contains(query, case=False)]
                    st.dataframe(df_filtered)
                elif cerca:
                    df_filtered = st.session_state['df_report_not_selected_folioManager'][st.session_state['df_report_not_selected_folioManager']['Report Names'].isin(cerca)]
                    st.dataframe(df_filtered)
                else:
                    df_filtered = st.session_state['df_report_not_selected_folioManager']
                    st.dataframe(st.session_state['df_report_not_selected_folioManager'])

                # Select Report Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                if st.button("Select", use_container_width=True):
                    st.session_state['df_report_selected_folioManager'] = pd.concat([st.session_state['df_report_selected_folioManager'], df_filtered], axis=0)
                    st.session_state['df_report_not_selected_folioManager'] = pd.merge(st.session_state['df_report_not_selected_folioManager'], df_filtered, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
                    st.session_state['isExpandedC'] = True
                    st.rerun()

            # Selected Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            with col2:
                st.text("Selected Reports")
                st.dataframe(st.session_state['df_report_selected_folioManager'])
                pul1, pul2 = st.columns(2)
                with pul1:
                    if st.button('Empty', use_container_width=True):
                        del st.session_state['df_report_selected_folioManager']
                        del st.session_state['df_report_not_selected_folioManager']
                        if 'df_report_to_compare' in st.session_state:
                            del st.session_state['df_report_to_compare']
                        st.rerun()

                with pul2:
                    if st.button('Compare', use_container_width=True):
                        if not st.session_state['df_report_selected_folioManager'].empty:
                            st.session_state['df_report_to_compare'] = st.session_state['df_report_selected_folioManager']['Report Names'].to_numpy()
                            st.session_state['isExpandedFM'] = False
                            st.rerun()
                        else:
                            st.warning('Error!')



         if 'df_report_to_compare' in st.session_state and len(st.session_state['df_report_to_compare']) > 0:

            data = []
            for report_name in st.session_state['df_report_to_compare']:
                df_symbol = myDB.get_df_report2(report_name)

                symbol = df_symbol.iloc[0, 4]
                matching_row = symbol_df[symbol_df.iloc[:, 0] == symbol]

                if not matching_row.empty:
                    data.append({'Strategy': report_name, 'Symbol': symbol, **matching_row.iloc[0, 1:].to_dict()})

                

            df_symbol = pd.DataFrame(data)





            app = option_menu(
                menu_title=None,
                options=['Rotational'],
                icons=[' '],
                menu_icon='chat-text-fill',
                orientation="horizontal",
                default_index=0,
                styles={
                    "container": {"padding": "20!important"},
                    "icon": {"color": "orange", "font-size": "25px"}, 
                    "nav-link": {"font-size": "20px", "text-align": "center", "margin":"1px"},
                }
                )
            
            if app == "Rotational":
                
                # CREO DIZIONARIO X OGNI STRATEGIA CON NET PROFIT, MAX DD, SYMBOLO E MERCATO PER FINESTRA TEMPORALE SCELTA
                def create_rotational_dictionary(file_list, df_symbol, window,  myDB):
                    portfolio_history = defaultdict(dict)
                    for file in file_list:
                        df_raw_report = myDB.get_df_report(file)
                        df_single = prepare_df_comparator(df_raw_report)
                        netProfits = df_single.resample(window)['Profit/Loss'].sum().astype("int32")
                        max_drawdowns = df_single.resample(window)['Drawdown'].max()
                        retOnAcc = ((netProfits / max_drawdowns) * 100)
                        
                        for date, profit in netProfits.items():
                            drawdown = max_drawdowns.loc[date]
                            returnonAccount = profit if drawdown == 0 else retOnAcc.loc[date]
                            strategy = os.path.basename(file)
                            portfolio_history[date][strategy] = {'Net Profit': profit, 'Drawdown': drawdown, 'Return On Account': returnonAccount}
                            
                            if strategy in df_symbol['Strategy'].values:
                                market = df_symbol.loc[df_symbol['Strategy'] == strategy, 'Markets'].values[0]
                                symbol = df_symbol.loc[df_symbol['Strategy'] == strategy, 'Symbol'].values[0]
                                margin = df_symbol.loc[df_symbol['Strategy'] == strategy, 'Overnight Margin'].values[0]
                            else:
                                market = "miss"
                                symbol = "miss"
                                margin = "miss"
                            portfolio_history[date][strategy]['Market'] = market
                            portfolio_history[date][strategy]['Symbol'] = symbol
                            portfolio_history[date][strategy]['Margin'] = margin

                    sorted_portfolio_history = {date: portfolio for date, portfolio in portfolio_history.items()}

                    return sorted_portfolio_history


                file_list = st.session_state['df_report_to_compare']
                resample_options = ['W', 'M', 'Q', 'A']
                window =st.selectbox('Select the sampling period:', resample_options, index=1)
                #  CREO UN DIZIONARIO CON METRICHE CALCOLATE SULLA FINESTRA TEMPORALE SCELTA
                portfolio_history = create_rotational_dictionary(file_list, df_symbol, window, myDB)
                #for date, portfolio in portfolio_history.items():
                #    st.write(f"{date}: {portfolio}")


                df_portfolio_history = pd.DataFrame.from_dict({(date, strategy): metrics 
                                                for date, portfolio in portfolio_history.items() 
                                                for strategy, metrics in portfolio.items()}, 
                                               orient='index')

                # Resetto gli indici per ottenere una struttura più pulita
                df_portfolio_history.reset_index(inplace=True)
                df_portfolio_history.rename(columns={'level_0': 'Date', 'level_1': 'Strategy'}, inplace=True)
                df_portfolio_history['Return On Account'] = df_portfolio_history['Return On Account']
                df_portfolio_history['Date'] = pd.to_datetime(df_portfolio_history['Date'], format="mixed", dayfirst=True)
                df_portfolio_history = df_portfolio_history.set_index('Date')

                
                



                # Definisci una funzione di filtro personalizzata con una lista di filtri su colonne multiple come parametro
                def custom_filter(group, filters):
                    for column, operator, value in filters:
                        # Applica il filtro corrente al DataFrame
                        if operator == '>':
                            group = group[group[column] > value]
                        elif operator == '>=':
                            group = group[group[column] >= value]
                        elif operator == '<':
                            group = group[group[column] < value]
                        elif operator == '<=':
                            group = group[group[column] <= value]
                        else:
                            raise ValueError("Invalid operator. Please provide one of the following: '>', '>=', '<', '<='.")
                    return group

                
                def convert_df_to_filter_list(df_filters):
                    filters = []
                    for index, row in df_filters.iterrows():
                        filter_name = row['Filter']
                        operator = row['Operator']
                        value = row['Value']
                        filters.append((filter_name, operator, value))
                    return filters

                # Utilizzo della funzione
                column_filter = [{"Filter": "", "Operator": "", "Value": "0"}]

                ### FILTRI SU STRATEGIE >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                df_fil_strategy = pd.DataFrame(column_filter)
                df_fil_strategy.Filter = df_fil_strategy.Filter.astype("category")
                df_fil_strategy.Filter = df_fil_strategy.Filter.cat.add_categories(("Net Profit", "Drawdown", "Return On Account"))
                df_fil_strategy.Operator = df_fil_strategy.Operator.astype("category")
                df_fil_strategy.Operator = df_fil_strategy.Operator.cat.add_categories((">", "<", "=", ">=", "<="))
                df_fil_strategy.Value = df_fil_strategy.Value.astype("int")

                with st.form("Filters"):
                    df_filters = st.data_editor(df_fil_strategy, hide_index=True, use_container_width=True, num_rows="dynamic")
                    submitted = st.form_submit_button("Submit", use_container_width=True)

                df_filters = df_filters.dropna()

                filters = convert_df_to_filter_list(df_filters)

                ## CREAZIONE PORTAFOGLIO GUIDA
                # Applica la funzione di filtro personalizzata al DataFrame raggruppato
                filtered_df = df_portfolio_history.groupby(['Date', 'Market']).apply(lambda x: custom_filter(x, filters))
                filtered_df.index = filtered_df.index.droplevel('Market')
                filtered_df.reset_index(level=0, inplace=True)
                filtered_df = filtered_df.drop(['Date'], axis=1)
                ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<




                st.write(df_portfolio_history)
                st.write(filtered_df)
                

                st.write("aaaaaaaaaaaa")

                                
            """
                data = [{"Type": "", "Filter": "", "Operator": "", "Value": "0",},]

                df_arr = pd.DataFrame(data)

                df_arr.Type = df_arr.Type.astype("category")
                df_arr.Type = df_arr.Type.cat.add_categories((
                    "Strategy", 
                    "Market", 
                    "Symbol", 
                    ))

                df_arr.Filter = df_arr.Filter.astype("category")
                df_arr.Filter = df_arr.Filter.cat.add_categories((
                    "Net Profit", 
                    "Drawdown", 
                    "Return On Account", 
                    "Strategies", 
                    ))

                df_arr.Operator = df_arr.Operator.astype("category")
                df_arr.Operator = df_arr.Operator.cat.add_categories((">", "<", "=", ">=", "<="))

                df_arr.Value = df_arr.Value.astype("int")

                with st.form("DataFilters"):
                    df_filters = st.data_editor(df_arr, hide_index=True, use_container_width=True, num_rows="dynamic")
                    submitted = st.form_submit_button("Submit", use_container_width=True)
                
                df_filters = df_filters.dropna()
                
                
                print(df_filters)
                print(df_portfolio_history)











            
                #### INPUT >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                col1, col2 = st.columns(2)
                with col1:
                    with st.form("INPUT FORM"):
                        resample_options = ['W', 'M', 'Q', 'A']
                        window =st.selectbox('Select the sampling period:', resample_options, index=1)
                        top_n = st.number_input('Num strategies for period', step=1, value=1, max_value=100)
                        max_strategies_per_market = st.number_input('Max strategies for market', step=1, value=1, max_value=100)
                        max_strategies_per_symbol = st.number_input('Max strategies for symbol', step=1, value=1, max_value=100)
                        submitted_input = st.form_submit_button("Submit")
                with col2:
                    filters = []
                    with st.form("FILTER FORM"):
                        # Aggiungi un widget multiselect per i filtri
                        filter_options = ['Drawdown', 'Net Profit', 'retOnAcc']
                        selected_filters = st.multiselect('Filter selection for single strategy', filter_options)
                        for filter_type in selected_filters:
                            filter_value = st.number_input(f'Inserisci il valore per {filter_type}', step=100)
                            filter_bool = st.checkbox(f'Seleziona il booleano per {filter_type}')
                            filters.append((filter_type, filter_value, filter_bool)) # Aggiungi il nuovo filtro alla lista
                        submitted_filter = st.form_submit_button("Submit")
                        if submitted_filter:
                            st.success(f'Filtri finali: {filters}')
                ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # CREO DIZIONARIO X OGNI STRATEGIA CON NET PROFIT, MAX DD, SYMBOLO E MERCATO PER FINESTRA TEMPORALE SCELTA
                def create_rotational_dictionary(file_list, df, window,  myDB):
                    portfolio_history = defaultdict(dict)
                    for file in file_list:
                        df_raw_report = myDB.get_df_report(file)
                        df_single = prepare_df_comparator(df_raw_report)
                        netProfits = df_single.resample(window)['Profit/Loss'].sum()
                        max_drawdowns = df_single.resample(window)['Drawdown'].max()
                        retOnAcc = ((netProfits / max_drawdowns) * 100)

                        for date, profit in netProfits.items():
                            
                            drawdown = max_drawdowns.loc[date]
                            if drawdown == 0:
                                returnonAccount = profit
                            else:
                                returnonAccount = retOnAcc.loc[date]
                            strategy = os.path.basename(file)
                            portfolio_history[date][strategy] = {'Net Profit': profit, 'Drawdown': drawdown, 'Return On Account': returnonAccount}
                            if strategy in df['Strategy'].values:
                                market = df.loc[df['Strategy'] == strategy, 'Markets'].values[0]
                                symbol = df.loc[df['Strategy'] == strategy, 'Symbol'].values[0]
                            else:
                                market = "miss"
                                symbol = "miss"
                            portfolio_history[date][strategy]['Market'] = market
                            portfolio_history[date][strategy]['symbol'] = symbol
                    # Ordina gli elementi del dizionario in base al netprofit
                    sorted_portfolio_history = {}
                    for date, portfolio in portfolio_history.items():
                        sorted_portfolio = dict(sorted(portfolio.items(), key=lambda item: item[1]['Net Profit'], reverse=True))
                        sorted_portfolio_history[date] = sorted_portfolio
                    return sorted_portfolio_history

                # FUNZIONE PER SELEZIONARE I SISTEMI IN BASE AI FILTRI
                def filter_portfolio(portfolio, filters, limit):
                    sorted_portfolio = list(portfolio.keys())
                    for key, filter_value, greater_than in filters:
                        reverse = True if key == 'Net Profit' or key == 'Return On Account' else False
                        sorted_portfolio = sorted(sorted_portfolio, key=lambda x: portfolio[x][key], reverse=reverse)
                        if filter_value is not None:
                            if greater_than:
                                sorted_portfolio = [system for system in sorted_portfolio if portfolio[system][key] >= filter_value]
                            else:
                                sorted_portfolio = [system for system in sorted_portfolio if portfolio[system][key] <= filter_value]
                    return sorted_portfolio[:limit]

                # CREO IL PORTAFOGLIO GUIDA PER SELEZIONARE I MIGLIORI SISTEMI x il prossimo mese IN BASE ALLA FUNZIONE FILTER PORTFOLIO
                def create_best_system_portfolio(sorted_portfolio_history, top_n, max_strategies_per_market, 
                                                max_strategies_per_symbol, filters, max_profit_per_market=None):
                    best_system_portfolio = {}
                    for date, portfolio in sorted_portfolio_history.items():
                        
                        best_systems = filter_portfolio(portfolio, filters, top_n)

                        market_count = defaultdict(int)
                        symbol_count = defaultdict(int)
                        market_profit = defaultdict(int)
                        selected_systems = []
                        for system in best_systems:
                            if 'Market' in portfolio[system] and 'symbol' in portfolio[system]: # NESSUNA SELEZIONE
                                market = portfolio[system]['Market']
                                symbol = portfolio[system]['symbol']
                                profit = portfolio[system]['Net Profit']
                                if market_count[market] < max_strategies_per_market and symbol_count[symbol] < max_strategies_per_symbol:
                                    if max_profit_per_market is None or market_profit[market] + profit <= max_profit_per_market:
                                        selected_systems.append(system)
                                        market_count[market] += 1
                                        symbol_count[symbol] += 1
                                        market_profit[market] += profit
                                if len(selected_systems) == top_n:
                                    break
                        best_system_portfolio[date] = {system: portfolio[system] for system in selected_systems}
                    return best_system_portfolio

                def create_portfolio_dataframe(best_system_portfolio, file_list, resample_day, myDB):
                    dataframes = {}
                    
                    for file in file_list:
                        df_raw_report = myDB.get_df_report(file)
                        df_single = prepare_df_comparator(df_raw_report)
                        dataframes[os.path.basename(file)] = df_single
                    
                    portfolio_df = pd.DataFrame()
                    
                    for date, portfolio in best_system_portfolio.items():
                        for system, profit in portfolio.items():
                            if system in dataframes:
                                system_df = dataframes[system].loc[date:date + pd.DateOffset(days=resample_day)].copy()
                                system_df['system'] = system
                                portfolio_df = pd.concat([portfolio_df, system_df])
                    
                    return portfolio_df
                                                


                if submitted_input:
                    file_list = st.session_state['df_report_to_compare']
                    #  CREO UN DIZIONARIO CON METRICHE CALCOLATE SULLA FINESTRA TEMPORALE SCELTA
                    rotational_portfolio = create_rotational_dictionary(file_list, df, window, myDB)
                    for date, portfolio in rotational_portfolio.items():
                        st.write(f"{date}: {portfolio}")
                    max_profit_per_market = 2000
                    best_system_portfolio = create_best_system_portfolio(rotational_portfolio, top_n, max_strategies_per_market, 
                                                                        max_strategies_per_symbol, filters, max_profit_per_market)
                    #for date, portfolio in best_system_portfolio.items():
                    #  st.write(f"{date}: {portfolio}")
                    # CREO IL DATAFRAME VERO E PROPRIO CON I SISTEMI RISULTATI DA BEST SYSTEM PORTFOGLIO
                    resample_days = {'W': 7, 'M': 30, 'Q': 90, 'A': 365}
                    resample_day = resample_days[window]
                    portfolio_df = create_portfolio_dataframe(best_system_portfolio, file_list, resample_day, myDB)
                    portfolio_df = portfolio_df.drop(['Cumulative P/L', 'Drawdown'], axis=1)
                    portfolio_df = portfolio_df.sort_values(by='ExitDate-Time')
                    portfolio_df['Cumulative P/L'] = portfolio_df['Profit/Loss'].cumsum() # MI CALCOLO L' EQUITY
                    portfolio_df['Max Equity'] = portfolio_df['Cumulative P/L'].cummax() # MAX EQUITY
                    portfolio_df['Drawdown']  = portfolio_df['Max Equity'] - portfolio_df['Cumulative P/L'] # DRAWDOWN DATAFRAME
                    portfolio_df['Drawdown %']  = round((portfolio_df['Max Equity'] - portfolio_df['Cumulative P/L']) / portfolio_df['Max Equity'] * 100,2) # DRAWDOWN % DATAFRAME
                    
                    
                    ### <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


                    fig2 = gen_equityDrawdown(portfolio_df, portfolio_df['Cumulative P/L'], portfolio_df['Drawdown'], portfolio_df['Drawdown %']) # EQUITY E DRAWDOWN  
                    st.plotly_chart(fig2, use_container_width=True)
            """





    
    else:
        st.error("In development")