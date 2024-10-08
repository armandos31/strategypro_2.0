import streamlit as st
import pandas as pd
from MyDB import MyDB
from graphs import *

def app():

    myDB = MyDB('armandinodinodello')

    if 'df_report_not_selected_repManager' not in st.session_state:
        tag_file_content = myDB.get_tag_file()
        reports_names = []
        reports_tags = []
        for entry in tag_file_content:
            reports_names.append(entry['file_name'])
            reports_tags.append(" ".join(entry['tags']))
        st.session_state['df_report_not_selected_repManager'] = pd.DataFrame({"Report Names":reports_names, "Tags":reports_tags})
        
        st.session_state['df_report_not_selected_comparator'] = st.session_state['df_report_not_selected_repManager']
        st.session_state['df_report_selected_comparator'] = pd.DataFrame({"Report Names": [], "Tags": []})
        if 'df_report_to_compare' in st.session_state:
            del st.session_state['df_report_to_compare']

        st.session_state['df_report_not_selected_strategyPro'] = st.session_state['df_report_not_selected_repManager']
        st.session_state['df_report_selected_strategyPro'] = pd.DataFrame({"Report Names": [], "Tags": []})
        if 'df_report_to_strategyPro' in st.session_state:
            del st.session_state['df_report_to_strategyPro']
                            
    if 'df_report_selected_repManager' not in st.session_state:
        st.session_state['df_report_selected_repManager'] = pd.DataFrame({"Report Names": [], "Tags": []})

    ### UPLOAD STRATEGY ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    with st.form("Upload", clear_on_submit=True):
        uploaded_files = st.file_uploader(label="Choose the file generated by your strategy", accept_multiple_files=True, type=["csv"])
        submitted = st.form_submit_button("Submit", use_container_width=True)
        if not not uploaded_files and submitted: 
            for uploaded_file in uploaded_files:
                myDB.insert_file(uploaded_file)
            del st.session_state['df_report_selected_repManager']
            del st.session_state['df_report_not_selected_repManager']
            st.rerun()

    col1, col2, col3 = st.columns(3)
    # Search Bar & Radio Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    with col1:
        cerca = st.multiselect('Select the reports to manage', st.session_state['df_report_not_selected_repManager'])
        query = st.text_input("Select Report")
        genre = st.radio( "FIlter bye", ["Name", "Tag"], index=0)    

        # Listener Search Bar ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # & All Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if query:
            filter = 'Report Names' if (genre == 'Name') else 'Tags'
            df_filtered = st.session_state['df_report_not_selected_repManager'][st.session_state['df_report_not_selected_repManager'][filter].str.contains(query, case=False)]
            st.dataframe(df_filtered)
        elif cerca:
            df_filtered = st.session_state['df_report_not_selected_repManager'][st.session_state['df_report_not_selected_repManager']['Report Names'].isin(cerca)]
            st.dataframe(df_filtered)
        else:
            df_filtered = st.session_state['df_report_not_selected_repManager']
            st.dataframe(st.session_state['df_report_not_selected_repManager'])

        # Select Report Btn ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if st.button("Select", use_container_width=True): 
            st.session_state['df_report_selected_repManager'] = pd.concat([st.session_state['df_report_selected_repManager'], df_filtered], axis=0)
            st.session_state['df_report_not_selected_repManager'] = pd.merge(st.session_state['df_report_not_selected_repManager'], df_filtered, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
            st.rerun()


    # Selected Report Table ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    with col2:
        st.text("Selected Reports")
        st.dataframe(st.session_state['df_report_selected_repManager'])
        if st.button('Empty', use_container_width=True):
            del st.session_state['df_report_selected_repManager']
            del st.session_state['df_report_not_selected_repManager']
            st.rerun()

    with col3:
        with st.form("Operation Form", clear_on_submit=True):
            st.text("Operation on selected reports:")
            tags_in = st.text_input("Tags")
            op_radio = st.radio( "Operation", ["Add Tags", "Remove Tags", "Delete Reports"], index=0)
            if st.form_submit_button("Ok", use_container_width=True):
                if op_radio == "Add Tags" or op_radio == "Remove Tags":
                    myDB.mod_tag(op_radio, tags_in.split(), st.session_state['df_report_selected_repManager'])
                else:
                    myDB.delete_report(st.session_state['df_report_selected_repManager'])
                query = ''
                del st.session_state['df_report_not_selected_repManager']
                del st.session_state['df_report_selected_repManager']
                st.rerun()