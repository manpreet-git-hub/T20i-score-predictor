import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

try:
    model = pickle.load(open('pipe.pkl', 'rb'))
except:
    st.warning("Model file not found. Please train the model using the notebook.")
    st.stop()


teams = ['Australia',
 'India',
 'Bangladesh',
 'New Zealand',
 'South Africa',
 'England',
 'West Indies',
 'Zimbabwe',
 'Pakistan',
 'Sri Lanka']

cities =['Colombo',  'Dubai',  'Johannesburg',  'Harare',  'Cape Town',  'Mirpur',  'Dhaka',  'Abu Dhabi',  'Sydney',  'Lahore',  'London',  'Auckland',  'Sylhet',  'Karachi',  'Wellington',  'Birmingham',  'Sharjah',  'St Lucia',  'Mumbai',  'Kuala Lumpur',  'Durban',  'Dublin',  'Christchurch',  'Mount Maunganui',  'Dambulla',  'Bridgetown',  'Windhoek',  'Centurion',  'Canberra',  'Melbourne',  'Hamilton',  'Barbados',  'Lauderhill',  'Southampton',  'Brisbane',  'Nottingham',  'Chattogram',  'Gros Islet',  'Guyana',  'Bulawayo',  'North Sound',  'Bristol',  'East London',  'Delhi',  'Perth',  'Hobart',  'Manchester',  'Kolkata',  'Chelmsford',  'Cardiff',  'Rawalpindi',  'Basseterre',  'Providence',  'Nagpur',  'Hambantota',  'Adelaide',  'Kandy',  'Bangalore',  'Trinidad',  'Potchefstroom',  'Chennai',  'Chittagong',  'Dunedin',  'Derby',  'Hangzhou',  'Chandigarh',  'Kingston',  'Nelson',  'Chester-le-Street',  'Antigua',  'Paarl',  'Tarouba',  'Gqeberha',  "St George's",  'Benoni',  'Navi Mumbai',  'Dharamsala',  'Ahmedabad',  'Taunton',  'Edinburgh',  'Guwahati',  'Rajkot',  'Lucknow',  'Kingstown',  'Napier',  'Brighton',  'Bangkok',  'Nairobi',  'Pune',  'Belfast',  'Northampton',  'Bready',  'Carrara',  'Accra',  'Doha',  'Coolidge',  'Loughborough',  'Queenstown',  'Gold Coast',  'Dallas',  'Khulna',  'Ranchi',  'Visakhapatnam',  'Surat',  'Thiruvananthapuram',  'Bloemfontein',  'Entebbe',  'Cave Hill',  'Multan',  'Gaborone',  'Indore',  'Hyderabad',  'Bengaluru',  'Jamaica',  'Cuttack',  'Canterbury',  'New York',  'Dundee',  'Port Elizabeth',  'Darwin',  'Al Amarat',  'King City',  'Leeds',  'St Kitts',  'Kimberley',  'Hove',  'St Vincent',  'The Hague',  'Mackay',  'Pietermaritzburg',  'Victoria',  'Geelong',  'Dominica']

st.title('Cricket Score Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city',sorted(cities))

col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input('Current Score')
with col4:
    overs = st.number_input('Overs done(works for over>5)')
with col5:
    wickets = st.number_input('Wickets out')

last_five = st.number_input('Runs scored in last 5 overs')

if st.button('Predict Score'):
    balls_left = 120 - (overs*6)
    wickets_left = 10 -wickets
    crr = current_score/overs

    input_df = pd.DataFrame(
     {'batting_team': [batting_team], 'bowling_team': [bowling_team],'city':city, 'current_score': [current_score],'balls_left': [balls_left], 'wickets_left': [wickets], 'crr': [crr], 'last_five': [last_five]})
    result = model.predict(input_df)
    st.header("Predicted Score - " + str(int(result[0])))


