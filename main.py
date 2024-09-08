import streamlit as st
from streamlit_option_menu import option_menu
from tools import Comparator, StrategyPro, Parameters, Report_Manager, Symbol_Manager, Account, Login_Register, Folio_Manager

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

if 'isGuest' not in st.session_state:
    st.session_state['isGuest'] = False

if st.session_state['isGuest']:
    tab_bar = ['Comparator', 'StrategyPro', 'Symbol Manager', 'Parameters', 'Report Manager']
else:
    tab_bar = ['Report Manager', 'Comparator', 'StrategyPro', 'Symbol Manager', 'Parameters', 'Folio Manager', 'Account']


if (('authentication_status' in st.session_state and st.session_state['authentication_status'] == True) and ('user_authorized' in st.session_state and st.session_state['user_authorized'])) or st.session_state['isGuest']:
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
    if app == "Account" and st.session_state['isGuest']:
        st.error("Warning! This feature is for subscriber customers only.")
    if app == "Account" and st.session_state['user_authorized']: 
        Account.app()

else:
    Login_Register.app()