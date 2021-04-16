import streamlit as st
import pandas as pd
import lightgbm as ltb
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.write("""
# Flight Price Prediction App
This app predicts the **Flight Fare**!
""")
st.write('---')

#Importing dataset
df =pd.read_csv('Flight Fare Prediction Model csv.csv')
X= df.drop('Price',axis=1)
Y= df['Price']



def user_input_features():
    st.write('### Departure Date')
    dep=st.date_input('-')

    journey_day=dep.day
    journey_month=dep.month

    st.write('### Arrival Date')
    arr =st.date_input('--')

    st.write('###  Stoppage')
    stops=st.selectbox('',('non-stop', '1 stop', '2 stops', '3 stops', '4 stops'))
    # Total Stops

    if (stops=='non-stop'):
        Total_Stops=0
    elif (stops=='1 stop'):
        Total_Stops=1
    elif (stops=='2 stops'):
        Total_Stops=2
    elif (stops=='3 stop'):
        Total_Stops=0
    else:
        Total_Stops=4


    st.write('### Choose Flight')
    option=st.selectbox('',('Jet Airways','IndiGo','Air India','Multiple carriers','SpiceJet','Vistara','Air Asia','GoAir'))
    # Choose Flight
    if (option=='Jet Airways'):
        Jet_Airways=1,
        IndiGo=0,
        Air_India=0,
        Multiple_carriers=0,
        SpiceJet=0,
        Vistara=0,
        Air_Asia =0,
        GoAir =0
    elif(option=='IndiGo'):
        Jet_Airways=0,
        IndiGo=1,
        Air_India=0,
        Multiple_carriers=0,
        SpiceJet=0,
        Vistara=0,
        Air_Asia =0,
        GoAir =0
    elif(option=='Air India'):
        Jet_Airways=0,
        IndiGo=0,
        Air_India=1,
        Multiple_carriers=0,
        SpiceJet=0,
        Vistara=0,
        Air_Asia =0,
        GoAir =0
    elif(option=='Multiple carriers'):
        Jet_Airways=0,
        IndiGo=0,
        Air_India=0,
        Multiple_carriers=1,
        SpiceJet=0,
        Vistara=0,
        Air_Asia =0,
        GoAir =0
    elif(option=='SpiceJet'):
        Jet_Airways=0,
        IndiGo=0,
        Air_India=0,
        Multiple_carriers=0,
        SpiceJet=1,
        Vistara=0,
        Air_Asia =0,
        GoAir =0
    elif(option=='Vistara'):
        Jet_Airways=0
        IndiGo=0
        Air_India=0
        Multiple_carriers=0
        SpiceJet=0
        Vistara=1
        Air_Asia =0
        GoAir =0
    elif(option=='Air Asia'):
        Jet_Airways=0
        IndiGo=0
        Air_India=0
        Multiple_carriers=0
        SpiceJet=0
        Vistara=0
        Air_Asia =1
        GoAir =0
    else:
        Jet_Airways=0
        IndiGo=0
        Air_India=0
        Multiple_carriers=0
        SpiceJet=0
        Vistara=0
        Air_Asia =0
        GoAir =1


    st.write('### From')
    source=st.selectbox('', ('Chennai','Delhi','Kolkata','Mumbai'))
    if (source=='Chennai'):
        Source_Chennai=1,
        Source_Delhi=0,
        Source_Kolkata=0,
        Source_Mumbai=0
    elif (source=='Delhi'):
        Source_Chennai=0,
        Source_Delhi=1,
        Source_Kolkata=0,
        Source_Mumbai=0
    elif (source =='Kolkata'):
        Source_Chennai=0,
        Source_Delhi=0,
        Source_Kolkata=1,
        Source_Mumbai=0
    else:
        Source_Chennai=0,
        Source_Delhi=0,
        Source_Kolkata=0,
        Source_Mumbai=1



    st.write('### To')
    destination= st.selectbox('',('Coachin','Delhi','Hyderabad','Kolkata','New Delhi'))
    if (destination== 'Coachin'):
        Destination_Cochin=1,
        Destination_Delhi=0,
        Destination_Hyderabad=0,
        Destination_Kolkata=0,
        Destination_New_Delhi=0
    elif (destination== 'Delhi'):
        Destination_Cochin=0,
        Destination_Delhi=1,
        Destination_Hyderabad=0,
        Destination_Kolkata=0,
        Destination_New_Delhi=0
    elif(destination== 'Hyderabad'):
        Destination_Cochin=0,
        Destination_Delhi=0,
        Destination_Hyderabad=1,
        Destination_Kolkata=0,
        Destination_New_Delhi=0
    elif(destination == 'Kolkata'):
        Destination_Cochin=0,
        Destination_Delhi=0,
        Destination_Hyderabad=0,
        Destination_Kolkata=1,
        Destination_New_Delhi=0
    else:
        Destination_Cochin=0,
        Destination_Delhi=0,
        Destination_Hyderabad=0,
        Destination_Kolkata=0,
        Destination_New_Delhi=1




    journey_day=dep.day
    journey_month=dep.month
    st.write('### Departure Time')

    Dep_Time_hour= st.slider( 'Hour',00,24,3)
    Dep_Time_min=  st.slider( 'Minute',0,60,12)
    st.write(Dep_Time_hour ,' : ', Dep_Time_min,' hrs')


    st.write('### Arrival Time')
    Arrival_Time_hour= st.slider('Hour', 00,24,10)
    Arrival_Time_min=  st.slider('Minute',0,60,20)
    st.write(Arrival_Time_hour ,' : ', Arrival_Time_min,' hrs')


    Duration_hours =  Arrival_Time_hour- Dep_Time_hour
    Duration_min   =  Arrival_Time_min- Dep_Time_min


    data={'Total_Stops':Total_Stops,
        'journey_day':journey_day,
        'journey_month':journey_month,
        'Dep_Time_hour':Dep_Time_hour,
        'Dep_Time_min':Dep_Time_min,
        'Arrival_Time_hour':Arrival_Time_hour,
        'Arrival_Time_min':Arrival_Time_min,
        'Duration_hours':Duration_hours,
        'Duration_min':Duration_min,
        'Jet_Airways':Jet_Airways,
        'IndiGo':IndiGo,
        'Air_India':Air_India,
        'Multiple_carriers':Multiple_carriers,
        'SpiceJet':SpiceJet,
        'Vistara':Vistara,
        'Air_Asia':Air_Asia,
        'GoAir':GoAir,
        'Destination_Cochin':Destination_Cochin,
        'Destination_Delhi':Destination_Delhi,
        'Destination_Hyderabad':Destination_Hyderabad,
        'Destination_Kolkata':Destination_Kolkata,
        'Destination_New_Delhi':Destination_New_Delhi,
        'Source_Chennai':Source_Chennai,
        'Source_Delhi':Source_Delhi,
        'Source_Kolkata':Source_Kolkata,
        'Source_Mumbai':Source_Mumbai

        }
    features = pd.DataFrame(data, index=[0])
    return features

data1=user_input_features()
st.subheader('User Input parameters')
st.write(data1)

import pickle
file=open('model.pkl','rb')
lgbm = pickle.load(file)

prediction = lgbm.predict(data1)

st.subheader('Your price for ticket is :')
st.write('Rs. ', prediction[0])








