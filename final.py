# Import necessary libraries
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from streamlit_option_menu import option_menu

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

# Define Streamlit app title and layout
st.set_page_config(layout='wide')

# Sidebar menu using external option_menu function
with st.sidebar:
    select = option_menu('Main Menu', ['Home','EDA Report','Prediction', 'Recommendation System'])

if select == 'Home':
    pass
if select=='EDA Report':
  pass    
  
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
            # Apply label encoding
            gender_encoded = label_encoder_gender.transform([gender])[0]
            contract_type_encoded = label_encoder_contract_type.transform([contract_type])[0]
            family_encoded = label_encoder_family.transform([family])[0]
            occupation_encoded = label_encoder_occupation.transform([occupation])[0]
            car_encoded = label_encoder_car.transform([own_car])[0]
            housetype_encoded = label_encoder_housetype.transform([housetype])[0]

            # Prepare input for prediction
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
    tfidf_matrix = pickle.load(open('/content/marix1.pkl', 'rb'))

  
    cosine_sim = pickle.load(open('/content/cosine.pkl', 'rb'))


    df_filtered = pickle.load(open('/content/df_filtered.pkl', 'rb'))
      
    st.title('Car Recommendation System')

      
    car_make = st.selectbox('Select a Car', df_filtered['Make'].unique())




    def get_recommendations(car_make, cosine_sim, df_filtered):
        
            
            filtered_df = df_filtered[df_filtered['Make'] == car_make]
            
            if filtered_df.empty:
                raise ValueError(f"Make '{car_make}' not found in dataset.")
            
            idx = filtered_df.index[0]
            sim_scores = list(enumerate(cosine_sim[idx]))

          
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            
            car_indices = [i[0] for i in sim_scores[1:101]]
            
            recommendations = df_filtered.iloc[car_indices]
            return recommendations


            


    if st.button('Show Recommendations'):
              recommendations = get_recommendations(car_make, cosine_sim, df_filtered)
              if not recommendations.empty:
                  st.subheader('Recommended Cars:')
                  st.table(recommendations[['Car Name', 'Model', 'Year', 'Price','Fuel','Transmission','Condition']])
              else:
                  st.error('No recommendations found.')
