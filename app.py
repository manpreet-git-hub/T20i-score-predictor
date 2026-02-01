import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

st.title('Cricket Score Predictor')

@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("data.csv")  
    except FileNotFoundError:
        st.warning("Training data file 'data.csv' not found.")
        st.stop()

    X = df.drop("target", axis=1)
    y = df["target"]

    model = Pipeline([
        ("regressor", LinearRegression())
    ])
    model.fit(X, y)
    
    with open("pipe.pkl", "wb") as f:
        pickle.dump(model, f)
    
    return model

try:
    model = pickle.load(open('pipe.pkl', 'rb'))
except FileNotFoundError:
    st.info("Model not found. Training a new model...")
    model = train_model()
    st.success("Model trained successfully!")

teams = ['Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 'England',
         'West Indies', 'Zimbawe', 'Pakistan', 'Sri Lanka']

cities = ['Colombo', 'Dubai', 'Johannesburg', 'Harare', 'Cape Town', 'Mirpur', 'Dhaka',
          'Abu Dhabi', 'Sydney', 'Lahore', 'London', 'Auckland', 'Sylhet', 'Karachi',
          'Wellington', 'Birmingham', 'Sharjah', 'St Lucia', 'Mumbai', 'Kuala Lumpur',
          'Durban', 'Dublin', 'Christchurch', 'Mount Maunganui', 'Dambulla', 'Bridgetown',
          'Windhoek', 'Centurion', 'Canberra', 'Melbourne', 'Hamilton', 'Barbados',
          'Lauderhill', 'Southampton', 'Brisbane', 'Nottingham', 'Chattogram', 'Gros Islet',
          'Guyana', 'Bulawayo', 'North Sound', 'Bristol', 'East London', 'Delhi', 'Perth',
          'Hobart', 'Manchester', 'Kolkata', 'Chelmsford', 'Cardiff', 'Rawalpindi',
          'Basseterre', 'Providence', 'Nagpur', 'Hambantota', 'Adelaide', 'Kandy',
          'Bangalore', 'Trinidad', 'Potchefstroom', 'Chennai', 'Chittagong', 'Dunedin',
          'Derby', 'Hangzhou', 'Chandigarh', 'Kingston', 'Nelson', 'Chester-le-Street',
          'Antigua', 'Paarl', 'Tarouba', 'Gqeberha', "St George's", 'Benoni', 'Navi Mumbai',
          'Dharamsala', 'Ahmedabad', 'Taunton', 'Edinburgh', 'Guwahati', 'Rajkot', 'Lucknow',
          'Kingstown', 'Napier', 'Brighton', 'Bangkok', 'Nairobi', 'Pune', 'Belfast',
          'Northampton', 'Bready', 'Carrara', 'Accra', 'Doha', 'Coolidge', 'Loughborough',
          'Queenstown', 'Gold Coast', 'Dallas', 'Khulna', 'Ranchi', 'Visakhapatnam', 'Surat',
          'Thiruvananthapuram', 'Bloemfontein', 'Entebbe', 'Cave Hill', 'Multan', 'Gaborone',
          'Indore', 'Hyderabad', 'Bengaluru', 'Jamaica', 'Cuttack', 'Canterbury', 'New York',
          'Dundee', 'Port Elizabeth', 'Darwin', 'Al Amarat', 'King City', 'Leeds', 'St Kitts',
          'Kimberley', 'Hove', 'St Vincent', 'The Hague', 'Mackay', 'Pietermaritzburg',
          'Victoria', 'Geelong', 'Dominica']

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select bowling team', sorted(teams))

city = st.selectbox('Select city', sorted(cities))

col3, col4, col5 = st.columns(3)
with col3:
    current_score = st.number_input('Current Score', min_value=0, step=1)
with col4:
    overs = st.number_input('Overs done (works for over>5)', min_value=0.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

last_five = st.number_input('Runs scored in last 5 overs', min_value=0, step=1)

if st.button('Predict Score'):
    if overs == 0:
        st.warning("Overs cannot be 0 for prediction!")
    else:
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets
        crr = current_score / overs

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [city],
            'current_score': [current_score],
            'balls_left': [balls_left],
            'wickets_left': [wickets_left],
            'crr': [crr],
            'last_five': [last_five]
        })

        result = model.predict(input_df)
        st.header("Predicted Score - " + str(int(result[0])))
