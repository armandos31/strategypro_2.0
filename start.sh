#!/usr/bin/bash

source new_env/bin/activate
nohup streamlit run main.py --server.headless=True > streamlit.log
