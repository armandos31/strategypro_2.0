import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def app():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    authenticator.login(max_concurrent_users=None) # Altrimenti il reset delle password non funziona, colpa di max_concurrent_users
    
    with st.sidebar:
        authenticator.logout()
        st.link_button("Manage Your Subscription", "https://billing.stripe.com/p/login/3cs8wE1yG8kJ5wsdQQ ")
    try:
        isPswChanged = authenticator.reset_password(st.session_state["username"])
        if isPswChanged:
            with open('config.yaml', 'w') as file:
                yaml.dump(config, file, default_flow_style=False)
                st.success('Password modified successfully')
    except Exception as e:
        st.error(e)