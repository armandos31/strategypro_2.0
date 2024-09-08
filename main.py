import streamlit as st
from streamlit_option_menu import option_menu
from tools import Comparator, StrategyPro, Parameters, Report_Manager, Symbol_Manager, Folio_Manager

try:
    st.set_page_config(layout="wide", page_title="StrategyPro", page_icon="logo_bordo.png")
except:
    print("Error")
    
PAGES_STYLE = """
<style> 
div[class^='block-container'] { padding-top: 2rem; } 
#MainMenu {visibility: hidden;} 
footer {visibility: hidden;} 
</style> """
st.markdown(PAGES_STYLE, unsafe_allow_html=True)

tab_bar = ['Report Manager', 'Comparator', 'StrategyPro', 'Symbol Manager', 'Parameters', 'Folio Manager']

app = option_menu(
        menu_title=None,
        options=tab_bar,
        icons=[' ', ' ', ' ', ' ', ' ', ' ', ' '],
        menu_icon='cast',
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "20!important"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "20px", "text-align": "center", "margin":"1px"},
        }
    )
if app == "Report Manager":
    Report_Manager.app()  
if app == "Symbol Manager":
    Symbol_Manager.app() 
if app == "Comparator":
    Comparator.app()      
if app == "StrategyPro":
    StrategyPro.app()
if app == "Parameters":
    Parameters.app()
if app == "Folio Manager":
    Folio_Manager.app()
