import stripe
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

def initState():
    if 'df_report_not_selected_repManager' in st.session_state:
        del st.session_state['df_report_not_selected_repManager']
    if 'df_report_selected_repManager' in st.session_state:
        del st.session_state['df_report_selected_repManager']
    if 'df_report_not_selected_strategyPro' in st.session_state:
        del st.session_state['df_report_not_selected_strategyPro']
    if 'df_report_selected_strategyPro' in st.session_state:
        del st.session_state['df_report_selected_strategyPro']
    if 'df_report_to_strategyPro' in st.session_state:
        del st.session_state['df_report_to_strategyPro']
    if 'df_report_not_selected_comparator' in st.session_state:
        del st.session_state['df_report_not_selected_comparator']
    if 'df_report_selected_comparator' in st.session_state:
        del st.session_state['df_report_selected_comparator']
    if 'df_report_to_compare' in st.session_state:
        del st.session_state['df_report_to_compare']

def check_subscription(config, username) -> bool:
    # Imposta la chiave segreta di Stripe
    stripe.api_key = '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'
    # Recupera la lista degli abbonati
    subscribers = stripe.Subscription.list()
    try:
        email = config['credentials']['usernames'][username]['email']
    except:
        st.write('User not registered in StrategyPro')
        return False
    for subscription in subscribers.data:
        customer_id = subscription.customer
        customer = stripe.Customer.retrieve(customer_id)
        # subscription.status può essere:
        # 'active', 'canceled', 'incomplete', 'incomplete_expired', 'past_due', 'paused', 'trialing', 'unpaid'
        if customer.email == email:
            if subscription.status == 'active':
                return True
            else:
                # Catturare gli altri stati della sottoscrizione per dare ulteriori info all'utente?
                return False
    # Se arriviamo qui significa che abbiamo listato tutto 'subscribers.data' senza trovare l'email dell'utente per cui si deduce che non è abbonato
    return False

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
    
    _, status, username = authenticator.login(max_concurrent_users=None)
    if status:
        st.session_state['user_authorized'] = check_subscription(config, username)
        if not st.session_state['user_authorized']:
            st.warning('Access Denied, subscription expired!')
            authenticator.logout()
        else:
            initState()
            st.rerun()
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your username and password')
    
    if st.button("Try It", use_container_width=True):
        st.session_state['isGuest'] = True
        initState()
        st.rerun()
            
    with st.expander('Sign in', expanded=False):
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(pre_authorization=True)
            if email_of_registered_user:
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
                    st.success('User registered successfully')
        except Exception as e:
            st.error(e)

        st.caption("* * * NOTE")
        st.caption("You can register only AFTER you have signed up for a SUBSCRIPTION")
        st.caption("If you purchased the subscription, please enter the email with which you made the purchase")


        
