import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from graphs import *
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_option_menu import option_menu
from MyDB import MyDB

def app():
    df = pd.DataFrame()
    
    file_list = []
    uploaded_file = st.file_uploader(label = "Select or drag one or more files to upload",
                                        accept_multiple_files = True, type = ["txt"])
    if uploaded_file:
        for file in uploaded_file:
            file_list.append(file.name)
    selected_file = st.selectbox("Select optimization report", file_list)
    if selected_file:
        try:
            for file in uploaded_file:
                if file.name == selected_file:
                    df = pd.read_csv(file, encoding="utf-16", sep='\t')
                    break
        except:
                st.error("Sorry, something went wrong") 
                st.stop() 

    if df.empty != True:
        grafico = option_menu(
                        menu_title=None,  # required
                        options=['2D Surface', '3D Surface', 'Parallel Coordinates', 'Linear Regression'],  # required
                        icons=[" ", " ", " "," ", " ", " ", " "],  # optional
                        menu_icon="cast",  # optional
                        default_index=0,  # optional
                        orientation="horizontal",
                    )
        if grafico == '2D Surface':
            default_x_col_name = df.columns[0]
            default_y_row_name = df.columns[1]
            with st.sidebar:
                x_col_name = st.selectbox("Select column for X-axis", options=list(df.columns), index=list(df.columns).index(default_x_col_name))
                y_row_name = st.selectbox("Select column for Y-axis", options=list(df.columns), index=list(df.columns).index(default_y_row_name))
                
            df = df.sort_values(by=[y_row_name, x_col_name])

            fig = px.scatter(df, 
                x=x_col_name, y=y_row_name, 
                color=x_col_name, color_continuous_scale='RdYlGn',
                hover_name=y_row_name, text=y_row_name,
                height=500)
            fig.update_traces(textposition='top center')  
            fig2 = px.bar(df, 
                x=y_row_name, y=x_col_name, 
                color=x_col_name, color_continuous_scale='RdYlGn',
                hover_name=y_row_name, text=y_row_name)
            

            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)



        if grafico == '3D Surface':
            # calculate the maximum value for each column of the dataframe
            default_x_col_name = df.columns[0]
            default_y_row_name = df.columns[1]
            default_z_row_name = df.columns[3]
            with st.sidebar:
                x_col_name = st.selectbox("Select column for X-axis", options=list(df.columns), index=list(df.columns).index(default_x_col_name))
                y_row_name = st.selectbox("Select column for Y-axis", options=list(df.columns), index=list(df.columns).index(default_y_row_name))
                z_name = st.selectbox("Select column for Z-axis", options=list(df.columns), index=list(df.columns).index(default_z_row_name))

            df = df.sort_values(by=[y_row_name, x_col_name])

            # convert to "set" to eliminate duplicates
            x_label = list(set(df[x_col_name]))
            y_label = list(set(df[y_row_name]))
            x_label.sort()
            y_label.sort()
            x_dim = len(x_label)
            y_dim = len(y_label)
            data = np.ndarray(shape=(y_dim, x_dim), dtype=float)
            df_number_of_row = df.shape[0]
            df_counter_row = 0
            # surface creation:
            for y_index in range(y_dim):
                dict_element = {}
                while y_label[y_index] == df.iloc[df_counter_row][y_row_name]:
                    dict_element[df.iloc[df_counter_row][x_col_name]] = df.iloc[df_counter_row][z_name]
                    df_counter_row += 1
                    if df_counter_row >= df_number_of_row:
                        break
                dict_len = len(dict_element)
                dict_counter = 0
                for x_index in range(x_dim):
                    if x_label[x_index] in dict_element:
                        data[y_index, x_index] = dict_element[x_label[x_index]]
                    else:
                        data[y_index, x_index] = 0
            figure = go.Figure(data=[go.Surface(x=x_label, y=y_label, z=data, hovertemplate=x_col_name + ": %{x}" + \
                                            "<br>" + y_row_name + ": %{y}" + \
                                            "<br>" + z_name + ": %{z}<extra></extra>")])
            figure.update_layout(scene=dict(xaxis_title=x_col_name, yaxis_title=y_row_name, zaxis_title=z_name),
                                width=750,
                                height=750)
            st.plotly_chart(figure, use_container_width=True)

        elif grafico == 'Parallel Coordinates':
            with st.sidebar:
                cols = st.multiselect("Select the columns to analyze", list(df.columns))

            if cols:
                selected_data = df[cols]
                with st.sidebar: master_col = st.selectbox("Select the Master column", list(cols))
                for col in selected_data.columns:
                    min_val = float(selected_data[col].min())
                    max_val = float(selected_data[col].max())
                    step_val = selected_data[col].std() / 10 
                    with st.sidebar: val_range = st.slider(f"Select range for {col}", min_val, max_val, (min_val, max_val), step=step_val) 
                    selected_data = selected_data.loc[(selected_data[col] >= val_range[0]) & (selected_data[col] <= val_range[1])]
                    
                fig = px.parallel_coordinates(selected_data, color=master_col, color_continuous_scale='Viridis')
                fig.update_layout(coloraxis_showscale=False,  height=850)          
                st.plotly_chart(fig, use_container_width=True)


    
        elif grafico == 'Linear Regression':
            with st.sidebar: 
                test_input = st.multiselect("Select independent variables (input)", list(df.columns))
                X = df[test_input]
                test_output = st.selectbox("Select the dependent variable (output)", list(df.columns))
                y = df[test_output]
                test_size = st.slider("Test Size % (0.1 to 0.9)", min_value=0.1, max_value=0.9, value=0.6, step=0.1)
            
            if test_input:
                ### LINEAR REGRESSION SETUP
                # Split the dataset into a training set and a testing set
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
                model = LinearRegression() # Create a linear regression model
                model.fit(X_train, y_train) # Train the model on the training dataset
                y_pred = model.predict(X_test) # Make predictions on the test dataset
                mse = mean_squared_error(y_test, y_pred) # Calculate the mean square error (MSE)
                r2 = r2_score(y_test, y_pred) * 100 # Calculate the coefficient of determination (R²)

                residuals = y_test - y_pred

                ### Graph
                col1, col2 = st.columns((1.5,4))
                ## DESCRIPTION OF SELECTED TABLES
                with col1:
                    st.write("Descriptive statistics of the variables:")
                    st.write(X.describe())
                
                ## CORRELATIONS BETWEEN THE SELECTED
                with col2:
                    correlation_matrix = X.corr()
                    st.write("Correlation matrix between variables:")
                    correlations_styler = correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format(precision=2)
                    st.dataframe(correlations_styler, use_container_width=True)

                    st.write("Coefficient of determination (R²): {:.2f}% ".format(r2))
                    st.write("Mean square error (MSE): {:.2f}".format(mse))
            
                residuals_fig = px.scatter(x=y_pred, y=residuals, labels={'x': 'Expected Values', 'y': 'Residuals'})
                residuals_fig.update_layout(title='Residuals Graph')
                st.plotly_chart(residuals_fig, use_container_width=True)

                # Create a forecast vs. actual value chart with Plotly
                prediction_vs_actual_fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Value', 'y': 'Expected Value'})
                prediction_vs_actual_fig.update_layout(title='Forecast Chart e. Actual Value')
                st.plotly_chart(prediction_vs_actual_fig, use_container_width=True)

                st.write("Coefficient of determination (R²): The R² value measures how well the model fits the data. A larger R² indicates that the model explains more of the variation in the output.")
                st.write("Mean Square Error (MSE): Mean square error is a measure of the error between model-predicted values and actual values. . The lower the MSE, the better the model fits the data. A high MSE value indicates that the model has relatively low prediction accuracy.")
    
    else:
        with st.sidebar:
            st.subheader('Upload your files')
