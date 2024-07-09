# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your trained model
model = pickle.load(open('/content/DTmodel21.pkl', 'rb'))

# Initialize LabelEncoders
label_encoder_gender = LabelEncoder()
label_encoder_contract_type = LabelEncoder()
label_encoder_family = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_car = LabelEncoder()
label_encoder_housetype = LabelEncoder()

# Define possible values for label encoders
gender_values = ['Male', 'Female']
contract_type_values = ['Cash', 'Revolving']
family_status_values = ['Married', 'Single']
occupation_values = ['Laborers', 'Core staff', 'Accountants', 'Managers', 'Drivers',
                     'Sales staff', 'Cleaning staff', 'Cooking staff', 'Security staff',
                     'Medicine staff', 'High skill tech staff', 'Private service staff',
                     'Low skill tech staff', 'Waiters/barmen staff', 'Realty agents',
                     'Secretaries', 'IT staff', 'HR staff']
car_values = ['Yes', 'No']
housetype_values = ['House / apartment', 'With parents', 'Municipal apartment',
                    'Rented apartment', 'Office apartment', 'Co-op apartment']

# Fit label encoders with possible values
label_encoder_gender.fit(gender_values)
label_encoder_contract_type.fit(contract_type_values)
label_encoder_family.fit(family_status_values)
label_encoder_occupation.fit(occupation_values)
label_encoder_car.fit(car_values)
label_encoder_housetype.fit(housetype_values)

st.set_page_config(layout='wide')

with st.sidebar:
    select = option_menu('Main Menu', ['Home','EDA Report','Prediction', 'Recommendation System'])

if select == 'Home':
    st.subheader('MODEL PERFORMANCE')
    dt = {
    'Model Name': ['LogisticRegression', 'Decision Tree', 'RandomForest'],
    'accuracy_score': [53.01, 93.69, 99.84],
    'recall_score': [60.34,94.64,99.75,],
    'f1_score': [54.04,93.70,99.85,]
      }

    pf = pd.DataFrame(dt)
    st.dataframe(pf)
    st.write('I prefer to use Decision Tree model in Bank risk controller systems')

if select=='EDA Report':

      # Load data
      conn = sqlite3.connect('loan.db')
      df = pd.read_sql_query("SELECT * FROM loan;", conn)
      conn.close()

      # Title of the app
      st.title('Exploratory Data Analysis of Loan Data')

      # Distplot for AMT_CREDIT_x
      st.header('Distribution of Loan Amounts (AMT_CREDIT_x)')
      fig1, ax1 = plt.subplots(figsize=(10, 6))
      sns.histplot(df['AMT_CREDIT_x'], kde=True, bins=30, ax=ax1)
      ax1.set_title('Distribution of Loan Amounts')
      ax1.set_xlabel('Loan Amount (AMT_CREDIT_x)')
      ax1.set_ylabel('Frequency')
      ax1.grid(True)
      st.pyplot(fig1)


      # Distplot for AMT_CREDIT_x
      st.header('Distribution of Loan Amounts (AMT_CREDIT_x)')
      fig2, ax2 = plt.subplots(figsize=(10, 6))
      sns.histplot(df['AMT_CREDIT_x'], kde=True, bins=30, ax=ax2)
      ax2.set_title('Distribution of Loan Amounts')
      ax2.set_xlabel('Loan Amount (AMT_CREDIT_x)')
      ax2.set_ylabel('Frequency')
      ax2.grid(True)
      st.pyplot(fig2)


      pca_cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT_x', 'AMT_ANNUITY_x', 'CNT_CHILDREN']
      df_pca = df[pca_cols].dropna()

      # Standardize the data
      scaler = StandardScaler()
      df_pca_scaled = scaler.fit_transform(df_pca)

      # Apply PCA
      pca = PCA(n_components=2)
      pca_result = pca.fit_transform(df_pca_scaled)

      # Create a DataFrame with the PCA results
      pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
      pca_df['TARGET'] = df['TARGET'].iloc[df_pca.index].values

      # Plot PCA results
      st.header('Dimensionality Reduction: PCA')
      fig3, ax3 = plt.subplots(figsize=(10, 6))
      sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='TARGET', alpha=0.7, ax=ax3)
      ax3.set_title('PCA of Loan Data')
      ax3.set_xlabel('Principal Component 1')
      ax3.set_ylabel('Principal Component 2')
      ax3.grid(True)
      st.pyplot(fig3)



      pairplot_cols = ['TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT_x', 'AMT_ANNUITY_x', 'CNT_CHILDREN', 'CODE_GENDER']
      df_pairplot = df[pairplot_cols]

      # Convert CODE_GENDER to numeric for the pair plot
      df_pairplot['CODE_GENDER'] = df_pairplot['CODE_GENDER'].map({'M': 0, 'F': 1})

      st.header('Multivariate Visualization: Pair Plot')
      fig4=sns.pairplot(df_pairplot, hue='TARGET', palette='Set2', diag_kind='kde', plot_kws={'alpha': 0.7})
      st.pyplot(fig4)




      st.header('Bivariate Visualization: Loan Amount vs Income')
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.scatterplot(data=df, x='AMT_INCOME_TOTAL', y='AMT_CREDIT_x', hue='TARGET', alpha=0.7, ax=ax)
      ax.set_title('Loan Amount vs Income')
      ax.set_xlabel('Income (AMT_INCOME_TOTAL)')
      ax.set_ylabel('Loan Amount (AMT_CREDIT_x)')
      ax.grid(True)
      st.pyplot(fig)



      st.header('Univariate Visualization: Distribution of Loan Amounts')
      fig, ax = plt.subplots(figsize=(10, 6))
      sns.histplot(df['AMT_CREDIT_x'], kde=True, bins=30, ax=ax)
      ax.set_title('Distribution of Loan Amounts')
      ax.set_xlabel('Loan Amount (AMT_CREDIT_x)')
      ax.set_ylabel('Frequency')
      ax.grid(True)
      st.pyplot(fig)



if select == 'Prediction':
    st.title('Bank Risk Controller Systems :money_with_wings:')

    col1, col2 = st.columns(2)
    with st.form('prediction form'):
        # Customer Details column
        with col1:
            st.header('Customer Details')
            ID_PREV = st.number_input('Previous Customer ID', min_value=0, max_value=999999999, step=1, value=0)
            gender = st.selectbox('Gender', gender_values)
            family = st.selectbox('Family Status', family_status_values)
            children = st.number_input('Number of Children', min_value=0, max_value=10, step=1, value=0)
            occupation = st.selectbox('Occupation Type', occupation_values)
            own_car = st.selectbox('Own Car', car_values)
            housetype = st.selectbox('House Type', housetype_values)

        # Loan Details column
        with col2:
            st.header('Loan Details')
            contract_type = st.selectbox('Contract Type', contract_type_values)
            income = st.number_input('Annual Income', min_value=0.0, max_value=10000000.0, step=1000.0, value=0.0)
            application = st.number_input('Application Amount', min_value=0.0, max_value=10000000.0, step=1000.0, value=0.0)
            downpayment = st.number_input('Down Payment', min_value=0.0, max_value=10000000.0, step=1000.0, value=0.0)
            credit = st.number_input('Credit Amount', min_value=0.0, max_value=10000000.0, step=1000.0, value=0.0)

        # Function to make predictions
        def predicts():
            
            gender_encoded = label_encoder_gender.transform([gender])[0]
            contract_type_encoded = label_encoder_contract_type.transform([contract_type])[0]
            family_encoded = label_encoder_family.transform([family])[0]
            occupation_encoded = label_encoder_occupation.transform([occupation])[0]
            car_encoded = label_encoder_car.transform([own_car])[0]
            housetype_encoded = label_encoder_housetype.transform([housetype])[0]

            
            user_input = {
                'CODE_GENDER': gender_encoded,
                'FLAG_OWN_CAR': car_encoded,
                'CNT_CHILDREN': children,
                'AMT_INCOME_TOTAL': income,
                'AMT_CREDIT_x': credit,
                'NAME_FAMILY_STATUS': family_encoded,
                'NAME_HOUSING_TYPE': housetype_encoded,
                'OCCUPATION_TYPE': occupation_encoded,
                'SK_ID_PREV': ID_PREV,
                'NAME_CONTRACT_TYPE_y': contract_type_encoded,
                'AMT_APPLICATION': application,
                'AMT_DOWN_PAYMENT': downpayment,

            }
            x = pd.DataFrame(user_input, index=[0])
            prediction = model.predict(x)




            ok= st.write(f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}")
            return ok



        # Button to trigger prediction
        st.form_submit_button('Predict', on_click=predicts)

if select == 'Recommendation System':
    file_path = 'Anonymized_Service_Use.xlsx'
    data = pd.read_excel(file_path)

    # Load the model and encoders
    model = load_model('recommendation_model_with_revenue.h5')
    with open('encoders_with_revenue.pkl', 'rb') as f:
        user_enc, service_enc, revenue_enc = pickle.load(f)

    # Streamlit app layout
    st.title("Service Recommendation System")

    # User inputs
    domain_list = data['Domain'].unique().tolist()
    selected_domain = st.selectbox("Select a Domain", domain_list)

    customer_list = data['Customer'].unique().tolist()
    selected_customer = st.selectbox("Select a Customer", customer_list)

    revenue_list = data['Customer Size Revenue'].unique().tolist()
    selected_revenue = st.selectbox("Select Customer Size Revenue", revenue_list)

    package_list = data['Package'].unique().tolist()
    selected_packages = st.multiselect("Select Packages", package_list)

    # Button to generate recommendations
    if st.button("Recommend Services"):
        # Encode user input
        customer_id = user_enc.transform([selected_customer])[0]
        revenue_id = revenue_enc.transform([selected_revenue])[0]

        
        filtered_data = data[(data['Domain'] == selected_domain) & (data['Package'].isin(selected_packages))]

        if not filtered_data.empty:
            
            service_ids = service_enc.transform(filtered_data['Service'].values)
            user_ids = np.full(len(service_ids), customer_id)
            revenue_ids = np.full(len(service_ids), revenue_id)

            
            predictions = model.predict([user_ids, service_ids, revenue_ids])
            predictions = predictions.flatten()

            
            top_service_indices = predictions.argsort()[-20:][::-1]
            top_services = service_enc.inverse_transform(service_ids[top_service_indices])

            
            unique_recommended_services = set()
            for service in top_services:
                unique_recommended_services.add(service)

            # Display recommendations
            st.write(f"Top service recommendations for {selected_customer} in {selected_domain} with revenue {selected_revenue} and packages {selected_packages}:")
            for i, service in enumerate(unique_recommended_services, 1):
                st.write(f"{i}. {service}")
        else:
            st.write("No services available for the selected domain and packages.")


