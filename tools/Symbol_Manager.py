import streamlit as st
from MyDB import MyDB
from function import *
from graphs import *

def app():
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    myDB = MyDB('armandinodinodello')
    symbol_df = myDB.get_symbol_list()

    col1, col2, col3 = st.columns(3)

    with col1: st.write("")
    with col2: 
        symbol_df = st.data_editor(symbol_df, num_rows="dynamic", use_container_width=True)
        if st.button('Save Symbol List', use_container_width=True):
            myDB.save_symbol_list(symbol_df)
            st.rerun()
    with col3: st.write("")

