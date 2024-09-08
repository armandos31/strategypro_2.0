import streamlit as st
from MyDB import MyDB
from function import *
from graphs import *

def app():
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    if st.session_state['isGuest']:
        user_id = ''
    else:
        user_id = st.session_state['username']

    myDB = MyDB(user_id, st.session_state['isGuest'])
    symbol_df = myDB.get_symbol_list()

    col1, col2, col3 = st.columns(3)

    if st.session_state['isGuest']:
        with col1: st.write("")
        with col2: 
            st.dataframe(symbol_df, use_container_width=True)
            st.write('* * * NOTE: If you want to add or edit the list you must purchase a subscription')
        with col3: st.write("")
    else:
        with col1: st.write("")
        with col2: 
            symbol_df = st.data_editor(symbol_df, num_rows="dynamic", use_container_width=True)
            if st.button('Save Symbol List', use_container_width=True):
                myDB.save_symbol_list(symbol_df)
                st.rerun()
        with col3: st.write("")

