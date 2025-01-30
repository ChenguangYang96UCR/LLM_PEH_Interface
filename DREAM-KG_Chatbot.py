import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime
import pgeocode
import pandas as pd
import openai
import re
import requests
from PIL import Image
from steamship import Steamship
from datetime import datetime
import numpy as np
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from datetime import datetime, time
from branca.element import IFrame
import random
# pip install "trubrics[streamlit]" for feedback
from trubrics.integrations.streamlit import FeedbackCollector
from streamlit_feedback import streamlit_feedback
import streamlit.components.v1 as components
from folium.plugins import AntPath
from urllib.request import urlopen
import json
import os
from py2neo import Graph

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import math
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
import fcntl
import streamlit_js_eval
import time
import utils
import pytz
import geopy

#from geopy.distance import geodesic

# ! Environment Variable
_RELEASE = False
huggingface_token = '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'

class PassengerDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.sequence_length].astype(np.float32)
        y = self.data[idx + self.sequence_length].astype(np.float32)
        return x, y

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size)
        self.i2o = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = nn.ReLU()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.relu(hidden)
        output = self.i2o(hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

def train(inputs, target):
    hidden = rnn.init_hidden(batch_size)  # initialize hidden
    output, hidden = rnn(inputs, hidden)  # forward pass
    target = target.view(batch_size, output_size)  # resize target to match the output
    loss = torch.sqrt(criterion(output, target))  # compute loss, user sqrt to take RMSE instead of MSE

    # backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def predict(inputs, target, hidden):
    rnn.eval()
    with torch.no_grad():
        output, hidden = rnn(inputs, hidden)  # forward pass
        target = target.view(batch_size, output_size)  # resize target to match the output

        return output, target

def my_component(key=None):
    component_value = _component_func(key=key, default=0)
    return component_value


# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, api_key, conversation_history):
    openai.api_key = api_key
    extraction_instruction = ("Extract the type of service, zipcode and time(include day of the week) from the following user query. "
                              "If the user doesn't provide a zipcode and only provides a general landmark, "
                              "please try to provide the corresponding zipcode for that landmark. "
                              "If user provide date is today, please check what day of the week today is. "
                              "If user doesn't provide date, please return None. "
                              "Please ensure that if the input is in Spanish, translate the input to English first, the responses are provided in English."
                              "Only provide the service type, the zipcode and the time period.")
    combined_query = f"{extraction_instruction}\n{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=full_conversation)
    
    if response.choices:
        conversation_history.append({"role": "user", "content": combined_query})
        conversation_history.append({"role": "assistant", "content": response.choices[0].message['content']})
    return response

# Function to classify service type
def classify_service_type(service_type, api_key):
    openai.api_key = api_key
    prompt = f"""
    Below are examples of service types with their categories:
    "Meal services": Food
    "Temporary housing": Shelter
    "Counseling services": Mental Health
    "Emergency food support": Food
    "Mental health counseling": Mental Health 

    Based on the examples above, classify the following service type into the correct category (Food, Shelter, Mental Health):
    "{service_type}": """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Updated to use the latest and more advanced model
        messages=[
            {"role": "system", "content": "You are a classifier that categorizes service types."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    # st.write(response)
    raw_category = response['choices'][0]['message']['content'].strip() if response['choices'] else "Other"
    # st.write("**Raw Model Response for Classification:**", raw_category) # make title as bold
    return raw_category


# Function for image analysis
def identify_image_geolocation(input_img, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Where is my location?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{input_img}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    r = response.json()
    print(r["choices"][0]["message"]["content"])
    return r

# Function to determine if current time is within a specified range
def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else: #Over midnight
        return nowTime >= startTime or nowTime <= endTime

# Function to safely convert time string to 24-hour format
def safe_convert_time(time_str):
    """Safely converts a time string to 24-hour format."""
    try:
        return datetime.strptime(time_str, '%I:%M %p').strftime('%H:%M')
    except ValueError:
        # Handle specific known issues
        if time_str == '0:00 AM':
            return '00:00'  # Convert '0:00 AM' to '00:00'
        elif time_str == '12:00 AM':  # Add other known issues as needed
            return '00:00'
        # Return original string or some default if no specific case matched
        return '00:00'  # Adjust this if necessary

@st.cache_data
def read_data(df):
    df = df.dropna(subset=['Latitude', 'Longitude'])  # Ensure we have the necessary location data
    # Get the current day and time
    now = datetime.now()
    current_day = now.strftime('%A')  # e.g., 'Monday'
    service_list = []
    data = []
    for index, row in df.iterrows():
        # Merge and clean services information
        main_services = eval(row['Main_Services']) if pd.notna(row['Main_Services']) else []
        other_services = eval(row['Other_Services']) if pd.notna(row['Other_Services']) else []
        services = main_services + other_services
        services = [service for service in services if service != 'None']
        
        # Get the opening hours for the current day
        opening_hours_today = row[current_day] if pd.notna(row[current_day]) else 'Unavailable'
        info = f"""
            <strong>{row['Service_Name']}</strong><br>
            <strong>Today's Hours:</strong> {opening_hours_today}<br>
            <strong>Services:</strong> {', '.join(services)}<br>
            <strong>Serving:</strong> {row['Serving']}<br>
            <strong>Phone Number:</strong> {row['Phone_Number']}<br>
            <strong>Eligibility:</strong> {row['Eligibility']}<br>
            <strong>Languages:</strong> {row['Languages']}<br>
            <strong>Cost:</strong> {row['Cost']}<br>
            <strong>Google Rating:</strong> {row['Google_Rating']}<br>
            <strong>Last Review:</strong> {row['Last_Review']}
        """

        # Append this information along with latitude and longitude
        data.append({
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'info': info
        })

        service_list.append([row['Service_Name'], row['Google_Rating']])

    return data, service_list

# read crime data
@st.cache_data
def read_crime_data(df):
    data = []
    for index, row in df.iterrows():
        # Format popup information to include today's hours
        info = f"""
            <strong>Crime Type:</strong> {row['text_general_code']}<br>
            <strong>Time:</strong> {row['dispatch_date_time']}<br>
            <strong>Geolocation:</strong> {row['location_block']}<br>
            <strong>Zipcode:</strong> {row['zipcode']}
        """

        # Append this information along with latitude and longitude
        data.append({
            'latitude': row['lat'],
            'longitude': row['lng'],
            'zipcode': row['zipcode'],
            'info': info
        })

    return data
# --------------

@st.cache_data
def translated_read_data(df):
    df = df.dropna(subset=['Latitude', 'Longitude'])  # Ensure we have the necessary location data
    # Get the current day and time
    now = datetime.now()
    current_day = now.strftime('%A')  # e.g., 'Monday'

    data = []
    for index, row in df.iterrows():
        # Merge and clean services information
        main_services = eval(row['Main_Services']) if pd.notna(row['Main_Services']) else []
        other_services = eval(row['Other_Services']) if pd.notna(row['Other_Services']) else []
        services = main_services + other_services
        services = [service for service in services if service != 'None']

        # Get the opening hours for the current day
        opening_hours_today = row[current_day] if pd.notna(row[current_day]) else 'Unavailable'
       # GoogleTranslator(source='auto', target='es').translate(str(opening_hours_today))

        # Format popup information to include today's hours
        info = f"""
            <strong>{row['Service_Name']}</strong><br>
            <strong>Horario de hoy:</strong> {GoogleTranslator(source='auto', target='es').translate(str(opening_hours_today))}<br>
            <strong>Servicios:</strong> {GoogleTranslator(source='auto', target='es').translate(', '.join(services))}<br>
            <strong>Servicio:</strong> {GoogleTranslator(source='auto', target='es').translate(str(row['Serving']))}<br>
            <strong>Número de teléfono:</strong> {row['Phone_Number']}<br>
            <strong>Elegibilidad:</strong> {GoogleTranslator(source='auto', target='es').translate(str(row['Eligibility']))}<br>
            <strong>Idiomas:</strong> {GoogleTranslator(source='auto', target='es').translate(str(row['Languages']))}<br>
            <strong>Costo:</strong> {GoogleTranslator(source='auto', target='es').translate(str(row['Cost']))}
            <strong>Calificación de Google:</strong> {GoogleTranslator(source='auto', target='es').translate(str(row['Google_Rating']))}
            <strong>Última revisión:</strong> {GoogleTranslator(source='auto', target='es').translate(str(row['Last_Review']))}
        """

        # Append this information along with latitude and longitude
        data.append({
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'info': info
        })

    return data

def parse_extracted_info(extracted_info):
    # Using regular expressions to find service type and zipcode robustly
    service_match = re.search(r"service(?: type)?:\s*(.+)", extracted_info, re.I)
    zipcode_match = re.search(r"zipcode:\s*(\d+)", extracted_info, re.I)
    weekday_match = utils.extract_weekday(extracted_info)
    # time_match = re.findall(r"\b(\d{2}):\d{2}\b", extracted_info)
    time_match = [int(hour) for hour in re.findall(r"\b(\d{2}):\d{2}\b", extracted_info)]

    service_type = service_match.group(1).title() if service_match else ""
    zipcode = zipcode_match.group(1) if zipcode_match else ""
    weekday = weekday_match if weekday_match else ""
    # Due to zipcode is number too
    if len(time_match) <= 1:
        service_time = 99
    else:
        service_time = time_match[1] if time_match else 99
    
    return service_type, zipcode, weekday, service_time

# * Send email to services
def send_email(language = 'en'):
    c1, c2 = st.columns([1, 1], gap="small")
    with c1:
        name_title = GoogleTranslator(source='auto', target=language).translate(str("Name *"))
        user_name = st.text_input(name_title)
    with c2:
        contract_title = GoogleTranslator(source='auto', target=language).translate(str("Email/Phone *"))
        user_contract = st.text_input(contract_title)
  
    Body = "Appointment name: " + user_name + '\n' + "User contract: " + user_contract + '\n'

    c1, c2, c3= st.columns([1, 1, 1], gap="small")
    with c1:
        address_title = GoogleTranslator(source='auto', target=language).translate(str("Street Address"))
        street_address = st.text_input(address_title)
    with c2:
        city_title = GoogleTranslator(source='auto', target=language).translate(str("City"))
        city = st.text_input(city_title)
    with c3:
        zip_title = GoogleTranslator(source='auto', target=language).translate(str("ZIP"))
        zip = st.text_input(zip_title)

    inqury_0 = GoogleTranslator(source='auto', target=language).translate(str("Please Select"))
    inqury_1 = GoogleTranslator(source='auto', target=language).translate(str("General Inquiry"))
    inqury_2 = GoogleTranslator(source='auto', target=language).translate(str("Adult Service"))
    inqury_3 = GoogleTranslator(source='auto', target=language).translate(str("Elderly Service"))
    inqury_4 = GoogleTranslator(source='auto', target=language).translate(str("Youth Service"))
    inqury_5 = GoogleTranslator(source='auto', target=language).translate(str("Family Service"))

    inquirys = [inqury_0, inqury_1, inqury_2, inqury_3, inqury_4, inqury_5]
    select_title = GoogleTranslator(source='auto', target=language).translate(str('I am inquiring about...'))
    selected_option = st.selectbox(select_title, inquirys)

    message_title = GoogleTranslator(source='auto', target=language).translate(str('Your Message'))
    message = st.text_input(message_title)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1], gap="large")
    with c4:
        appointment_title = GoogleTranslator(source='auto', target=language).translate(str("Make Appointment"))
        appointment = st.button(appointment_title)
    if appointment:
        if inquirys != inqury_0:
            Body = Body + "Inquiry: " + selected_option + '\n' + 'Message: ' + message + '\n'
        if street_address !=  "":
            Body = Body + "Street: " + street_address + '\n'
        if city !=   "":
            Body = Body + "City: " + city + '\n'
        if zip !=  "":
            Body = Body + "Zip: " + zip + '\n'

        send_email_spinner = GoogleTranslator(source='auto', target=language).translate(str("Sending e-mail, please wait ..."))
        with st.spinner(send_email_spinner):
            try:
                msg = MIMEText(Body)
                msg['From'] = "ucr.dreamkg@gmail.com"
                msg['To'] = "ucr.dreamkg@gmail.com"
                msg['Subject'] = 'Appointment'
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login("ucr.dreamkg@gmail.com", 'vdvo hgpq tdba fiai')
                server.sendmail("ucr.dreamkg@gmail.com", "ucr.dreamkg@gmail.com", msg.as_string())
                server.quit()
                st.success('Email sent successfully! 🚀')
            except Exception as e:
                st.error(f"Error: Can't send e-mail : {e}")

#! main function of interface 
if __name__ == '__main__':
    if not _RELEASE:
        _component_func = components.declare_component(
            "my_component",
            #url="http://localhost:3001",
            url="http://localhost:8501/",
        )
    else:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.join(parent_dir, "frontend/build")
        _component_func = components.declare_component("my_component", path=build_dir)

    # Set logger to record log 
    logger =  utils.set_logger()

    if 'mainpageId' not in st.session_state:
        st.session_state.mainpageId = "False"
    # Streamlit UI
    img = Image.open('dream_kg_logo_v2.png')
    st.image(img)

    # * Acquire the location of user
    loc = streamlit_js_eval.get_geolocation()
    # st.write(f"Your coordinates are {loc}")

    #st.markdown("# **Start Here ↓**", icon="👇")
    st.info("**Welcome to DREAM-KG chatbot, start here ↓**", icon="👋") #edited: 10/24
    st.markdown("### 💬 Ask me about services")
    query_txt = "Enter your query: Find me a food pantry near market east && families in Philadelphia.\n(Ingrese su consulta: Búsqueme una despensa de alimentos cerca de Market East && Family en Filadelfia.)"
    user_query = st.text_input(query_txt, key="user_query")

    # Service audience checkbox
    c1, c2, c3, c4, c5= st.columns([1, 1, 1, 1, 1], gap="small")
    with c1:
        adult_check = st.checkbox('adults 18+', value=False, key='adults 18+')
    with c2:
        families_check = st.checkbox('families', value=False, key='families')
    with c3: 
        AllAges_check = st.checkbox('all ages', value=False, key='all ages')
    with c4: 
        ptsd_check = st.checkbox('ptsd', value=False, key='ptsd')
    with c5:
        veterans_check = st.checkbox('veterans', value=False, key='veterans')
    
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1], gap="small")
    with c1:
        emergency_check = st.checkbox('emergency', value=False, key='emergency')
    with c2:
        individuals_check = st.checkbox('individuals', value=False, key='individuals')

    if adult_check or families_check or AllAges_check or ptsd_check or veterans_check or emergency_check or individuals_check:
        audience_select = True
    
    # Submit button
    submit_button = st.button("Submit", type="primary")
    if submit_button:
        st.session_state.mainpageId = "True"
    #submit_button = st.form_submit_button("Submit", type="primary", use_container_width=True)

    replicate_text = "DREAM-KG: Develop Dynamic, REsponsive, Adaptive, and Multifaceted Knowledge Graphs to address homelessness with Explainable AI"
    replicate_link = "https://dreamkg.com/"
    replicate_logo = "https://storage.googleapis.com/llama2_release/Screen%20Shot%202023-07-21%20at%2012.34.05%20PM.png"
    st.markdown(
                ":orange[**Resources:**]  \n"
                f"<img src='{replicate_logo}' style='height: 1em'> [{replicate_text}]({replicate_link})",
                unsafe_allow_html=True
    )

    ### 10/24 - edited
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #fcb290;
        }
    </style>
    """, unsafe_allow_html=True)

    # Initialize global variables
    #! Openai api key
    conversation_history = []
    api_key = ''
        
    if st.session_state.mainpageId == "True":
        if user_query is '':
            st.write('Your input is empty, please check your query!')
            st.stop()
        input_language = detect(user_query)
        # translation
        logger.debug("user_query:" + user_query)
        translated_user_query = GoogleTranslator(source='auto', target='en').translate(str(user_query))

        response = ask_openai_for_service_extraction(translated_user_query, api_key, conversation_history)
        logger.debug("translated_user_query:" + translated_user_query)
       
        if response.choices:
            extracted_info = response.choices[0].message['content'].strip()
            # st.write("Extracted Information:", extracted_info)
            service_type, zipcode, weekday, service_time = parse_extracted_info(extracted_info)
            #print("service_type:", service_type)
            # crime incidents analysis
            crime_data_df = pd.read_csv('Final_Philadelphia_Crime_Data_2023.csv')
            crime_data = crime_data_df.values
            number_of_crimes = np.where(int(zipcode) == crime_data[:, 18])[0].shape[0]
            # category of crimes
            number_of_crimes_index = np.where(int(zipcode) == crime_data[:, 18])[0]
            sub_crime_data = crime_data[number_of_crimes_index,]
            # property victimization
            number_of_property_crimes_index = \
            np.where(('Thefts' == sub_crime_data[:, 13]) | ('Vandalism/Criminal Mischief' == sub_crime_data[:, 13])
                    | ('Arson' == sub_crime_data[:, 13]) | ('Burglary Residential' == sub_crime_data[:, 13])
                    | ('Theft from Vehicle' == sub_crime_data[:, 13]) | (
                                'Motor Vehicle Theft' == sub_crime_data[:, 13]))[0]
            number_of_property_crimes = number_of_property_crimes_index.shape[0]
            # personal victimization
            number_of_personal_crimes_index = \
            np.where(('Other Assaults' == sub_crime_data[:, 13]) | ('Robbery No Firearm' == sub_crime_data[:, 13])
                    | ('Robbery Firearm' == sub_crime_data[:, 13]) | (
                                'Offenses Against Family and Children' == sub_crime_data[:, 13])
                    | ('Other Sex Offenses (Not Commercialized)' == sub_crime_data[:, 13]))[0]
            number_of_personal_crimes = number_of_personal_crimes_index.shape[0]
            crime_frequency = crime_data_df.loc[
                np.where(int(zipcode) == crime_data[:, 18])[0], 'text_general_code'].value_counts().to_dict()

            # Get the eastern current time 
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(tz=eastern)
            current_time = now.strftime("%H:%M:%S")
            weekday_name = now.strftime('%A')
            current_hour = now.hour

            if service_type and zipcode:
                try:
                    classified_service_type = classify_service_type(service_type, api_key)
                    time_title = "#### Current Time in Eastern Standard Time"
                    time_markdown = GoogleTranslator(source='auto', target=input_language).translate(str(time_title))
                    st.markdown(time_markdown)
                    st.write(weekday_name + ' ' + current_time)
                    
                    service_title = "#### Type of Service"
                    service_markdown = GoogleTranslator(source='auto', target=input_language).translate(str(service_title))
                    st.markdown(service_markdown)
                    st.write(classified_service_type)

                    if classified_service_type == 'Shelter':
                        st.write("**Specific Temporary Housing for Veteran:**", "If you are veteran, please consider Veterans Multi Service Center (Phone: 215-238-8067; Address: 213-217 N 4th St, Philadelphia, PA 19106)")
                        st.write("**Specific Temporary Housing for Single Woman/Women:**", "If you are single woman/women, please consider House of Passage (Phone: 267-713-7778; Address: 111 N 49th St, Philadelphia, PA 19139)")
                        st.write("**Specific Temporary Housing for Single Man/Men:**", "If you are single man/men, please consider Mark Hinson Resource Center (Phone: 215-923-2600; Address: 1701 W Lehigh Ave, Philadelphia, PA 19132")
                        st.write("**Specific Temporary Housing for Families:**", "If you have families, please consider Salvation Army Red Shield Center (Phone: 215-787-2887; Address: 715 N Broad St, Philadelphia, PA 19123")
                    
                    zipcode_title = "#### Zipcode"
                    service_markdown = GoogleTranslator(source='auto', target=input_language).translate(str(zipcode_title))
                    st.markdown(service_markdown)
                    st.write(zipcode)

                    if classified_service_type != "Other":
                        service_files = {
                            "Shelter": "Final_Temporary_Shelter_20250111.csv",
                            "Mental Health": "Final_Mental_Health_20250111.csv",
                            "Food": "Final_Emergency_Food_20250111.csv"
                        }
                        print('before service files')
                        if classified_service_type != "Shelter" and classified_service_type != "Mental Health" and classified_service_type != "Food":
                            print('in service files')
                            service_type_warning = 'Service type is not recognized. Please try again with a different service type. Such that: "Shelter", "Mental Health", "Food". And there is the example for query: Find me a food pantry near market east && families in Philadelphia.'
                            service_type_waring_trans = GoogleTranslator(source='auto', target=input_language).translate(str(service_type_warning))
                            st.markdown('''##### :red['''+ service_type_waring_trans + ''']''')
                            st.stop()
                        print('before read csv')
                        datafile = service_files[classified_service_type]
                        df = pd.read_csv(datafile)
                        data, service_list = read_data(df)
                        # load crime data (till 07/2024) for visualization
                        crime_data_df = pd.read_csv('three_days_philly_incidents_with_zipcode_2025.csv')
                        crime_data = read_crime_data(crime_data_df)

                        crime_information_title = "#### Crimes Information near " + str(zipcode)
                        crime_information_markdown = GoogleTranslator(source='auto', target=input_language).translate(str(crime_information_title))
                        st.markdown(crime_information_markdown)
                        filter_crimes = utils.filter_crime_based_zipcode(crime_data, zipcode)
                        utils.get_crimes_summary(filter_crimes, st, input_language)

                        morning_crime_data_df = pd.read_csv("three_days_philly_incidents_2025_morning.csv")
                        morning_crime_data = read_crime_data(morning_crime_data_df)

                        afternoon_crime_data_df = pd.read_csv("three_days_philly_incidents_2025_afternoon.csv")
                        afternoon_crime_data = read_crime_data(afternoon_crime_data_df)

                        evening_crime_data_df = pd.read_csv("three_days_philly_incidents_2025_evening.csv")
                        evening_crime_data = read_crime_data(evening_crime_data_df)

                        crime_df = pd.read_csv('Final_Pandas_tensor_2023.csv')
                        crime_df.columns = ["month", "zipcode", "Homicide Criminal", "Rape", "Robbery No Firearm",
                                            "Aggravated Assault No Firearm", "Burglary Residential",
                                            "Thefts", "Motor Vehicle Theft", "All Other Offenses", "Other Assaults",
                                            "Forgery and Counterfeiting", "Fraud", "Embezzlement",
                                            "Receiving Stolen Property",
                                            "Vandalism/Criminal Mischief", "Weapon Violations",
                                            "Prostitution and Commercialized Vice", "Other Sex Offenses",
                                            "Narcotic/Drug Law Violations", "Gambling Violations",
                                            "Offenses Against Family and Children", "DRIVING UNDER THE INFLUENCE",
                                            "Liquor Law Violations", "Public Drunkenness", "Disorderly Conduct",
                                            "Vagrancy/Loitering", "Theft from Vehicle",
                                            "psa_1", "psa_2", "psa_3", "psa_4", "psa_A",
                                            "total_hours"]  # change col name
                        print('transfer zipcode')
                        zipcode_num = int(zipcode)
                        logger.debug("zipcode_num is:" + zipcode)
                        # top-3 crime incidents
                        crime_type_list = sorted(crime_frequency, key=crime_frequency.get, reverse=True)[:3] #'All_Other_Offenses' # only focus on top three crime types
                        final_pred_res = [] # number of prediction in weeks
                        print(crime_type_list)
                        for crime_type in crime_type_list:
                            new_crime_df = crime_df.loc[crime_df['zipcode'] == zipcode_num, ['month', crime_type]]
                            # print(crime_df.query('zipcode' = zipcode_num))
                            passenger_counts = new_crime_df[crime_type].values
                            sequence_length = 3  # we will use data of 12 months to predict the passenger in 13th month - need to change
                            batch_size = 1
                            dataset = PassengerDataset(passenger_counts, sequence_length)
                            test_size = 3  # 12 months for test
                            train_size = len(dataset) - test_size
                            train_dataset = Subset(dataset, range(0, train_size))
                            test_dataset = Subset(dataset, range(train_size, len(dataset)))
                            assert len(train_dataset) + len(test_dataset) == len(dataset)
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                            input_size = sequence_length
                            output_size = 1  # predict 1 month
                            hidden_size = 32
                            rnn = RNN(input_size, hidden_size, output_size)
                            num_epochs = 50
                            learning_rate = 0.0002
                            criterion = nn.MSELoss()
                            optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
                            print_step = 20
                            all_losses = []
                            for epoch in range(num_epochs):
                                loss_this_epoch = []
                                for inputs, target in train_loader:
                                    loss = train(inputs, target)
                                    loss_this_epoch.append(loss.item())
                                loss_this_epoch = np.array(loss_this_epoch).mean()
                                all_losses.append(loss_this_epoch)

                            y_true = []
                            y_pred = []

                            hidden = rnn.init_hidden(batch_size)
                            for inputs, target in test_loader:
                                output, target = predict(inputs, target, hidden)
                                y_pred.append(output.item())
                                y_true.append(target.item())

                            y_true = (np.array(y_true))
                            y_pred = (np.array(y_pred))
                            print("y_pred:", y_pred)
                            final_pred_res.append(np.floor(y_pred))

                        chart_data = pd.DataFrame(np.array(final_pred_res).transpose(), columns=crime_type_list)
                        chart_data['Day'] = [GoogleTranslator(source='auto', target=input_language).translate(str("Day 1")), 
                                             GoogleTranslator(source='auto', target=input_language).translate(str("Day 2")), 
                                             GoogleTranslator(source='auto', target=input_language).translate(str("Day 3"))]
                        # chart_data['Day'] = ["Day 1", "Day 2", "Day 3"]
                        translate_crime_type = [""]*len(crime_type_list)
                        for index in range(len(crime_type_list)):
                            translate_crime_type[index] = GoogleTranslator(source='auto', target=input_language).translate(crime_type_list[index])
                            
                        st.scatter_chart(
                            chart_data,
                            x="Day",
                            y=crime_type_list,
                            #size="col4",
                            color=["#fd0", "#f0f", "#04f"],  # Optional
                        )

                        # Use pgeocode for geocoding
                        nomi = pgeocode.Nominatim('us')
                        location_info = nomi.query_postal_code(zipcode)

                        if not location_info.empty:
                            latitude_user = location_info['latitude']
                            longitude_user = location_info['longitude']
                            print("latitude_user, longitude_user:", latitude_user, longitude_user)
                            city_name = location_info['place_name']
                            # client = Steamship(api_key="25FDC915-9156-4BFB-BA9B-1B213DF1E699")

                            extract_services = []
                            top_services = ["KITHS Kitchen and Garden (KITHS)", "Social Services -Basic Needs Assistance (Helping Hands Ministry Inc)", "Emergency Housing for Veterans (Fresh Start Foundation)",\
                                            "Adult Behavioral Health Inpatient Treatment (Friends Hospital)", "Adult Outpatient Services (Hispanic Community Counseling Services)", "Opioid Treatment Program (Achievement Through Counseling and Treatment)",\
                                            "Church-Based Shelters (Bethesda Project)", "RHD Fernwood Program (Resources for Human Development-Pennsylvania)", "Various Community Events and Programs (Conquerors Community Development Corporation)"]                            
                            for service in service_list:
                                if service[0] in top_services:
                                    start_brasket = service[0].find('(')
                                    end_brasket = service[0].find(')', start_brasket + 1)
                                    service_name = service[0]
                                    extract_services.append([service_name[start_brasket+1:end_brasket], service[1]])

                            # ! Service information
                            service_header = "Services Information"
                            serviceheader_markdown = GoogleTranslator(source='auto', target=input_language).translate(str(service_header))
                            st.header(serviceheader_markdown)
                            option_services = []
                            print('extract services list : {0}'.format(len(extract_services)))
                        
                            if audience_select:
                                time_services = []
                                select_audience = []
                                if adult_check:
                                    select_audience.append('adult')
                                if families_check:
                                    select_audience.append('families')
                                if ptsd_check:
                                    select_audience.append('ptsd')
                                if AllAges_check:
                                    select_audience.append('all ages')
                                if emergency_check:
                                    select_audience.append('emergency')
                                if individuals_check:
                                    select_audience.append('individuals')
                                if veterans_check:
                                    select_audience.append('veterans')
                                audience_services = utils.get_serving_from_list(select_audience, logger)
                                if not weekday == "":
                                    # * Can get weekday from user's question
                                    if service_time == 99:
                                        time_services = utils.get_services_time(weekday, current_hour, logger)
                                    else:
                                        time_services = utils.get_services_time(weekday, service_time, logger)
                                else:
                                    # * Can not get weekday from user's question
                                    if service_time == 99:
                                        time_services = utils.get_services_time(weekday_name, current_hour, logger)
                                    else:
                                        time_services = utils.get_services_time(weekday_name, service_time, logger)
                                audience_services.extend(time_services)
                                duplicate_services = utils.get_duplicate_service_name(audience_services)
                                
                                extract_duplicate_services = []
                                for service in service_list:
                                    if service[0] in duplicate_services:
                                        start_brasket = service[0].find('(')
                                        end_brasket = service[0].find(')', start_brasket + 1)
                                        service_name = service[0]
                                        extract_duplicate_services.append([service_name[start_brasket+1:end_brasket], service[1]])
                                service_info_spinner = 'Loading service information, please wait ...'
                                service_info_spinner_trans = GoogleTranslator(source='auto', target=input_language).translate(str(service_info_spinner))
                                with st.spinner(service_info_spinner_trans):
                                    if len(extract_duplicate_services) == 0:
                                        option_services = extract_services
                                        service_information = utils.getQuestion_answer(extract_services, st, input_language)
                                    else:
                                        option_services = extract_duplicate_services
                                        service_information = utils.getQuestion_answer(extract_duplicate_services, st, input_language)
                                
                            else:
                                time_services = []
                                if not weekday == "":
                                    # * Can get weekday from user's question
                                    if service_time == 99:
                                        time_services = utils.get_services_time(weekday, current_hour, logger)
                                    else:
                                        time_services = utils.get_services_time(weekday, service_time, logger)
                                else:
                                    # * Can not get weekday from user's question
                                    if service_time == 99:
                                        time_services = utils.get_services_time(weekday_name, current_hour, logger)
                                    else:
                                        time_services = utils.get_services_time(weekday_name, service_time, logger)
                                    
                                    extract_time_services = []
                                    for service in service_list:
                                        if service[0] in time_services:
                                            start_brasket = service[0].find('(')
                                            end_brasket = service[0].find(')', start_brasket + 1)
                                            service_name = service[0]
                                            extract_time_services.append([service_name[start_brasket+1:end_brasket], service[1]])
                                    # print(extract_time_services)
                                    service_info_spinner = 'Loading service information, please wait ...'
                                    service_info_spinner_trans = GoogleTranslator(source='auto', target=input_language).translate(str(service_info_spinner))
                                    with st.spinner(service_info_spinner_trans):
                                        if len(extract_time_services) == 0:
                                            option_services = extract_services
                                            service_information = utils.getQuestion_answer(extract_services, st, input_language)
                                        else:
                                            option_services = extract_time_services
                                            service_information = utils.getQuestion_answer(extract_time_services, st, input_language)
                            print("Before map filter.")
                            time_zone_select = 'Select which time you prefer, then we will give you a crime map at that time.'
                            time_zone_select_markdown = GoogleTranslator(source='auto', target=input_language).translate(str(time_zone_select))
                            st.markdown('''##### :red['''+ time_zone_select_markdown + ''']''')
                            # st.write("Select which time you prefer, then we will give you a crime map at that time.")
                            c1, c2, c3= st.columns([1, 1, 1], gap="small")
                            with c1:
                                morning_trans = GoogleTranslator(source='auto', target=input_language).translate(str('Morning'))
                                Morning = st.checkbox(morning_trans, value=False, key='Morning')
                            with c2:
                                afternoon_trans = GoogleTranslator(source='auto', target=input_language).translate(str('Afternoon'))
                                Afternoon = st.checkbox(afternoon_trans, value=False, key='Afternoon')
                            with c3:
                                evening_trans = GoogleTranslator(source='auto', target=input_language).translate(str('Evening'))
                                Evening = st.checkbox(evening_trans, value=False, key='Evening')
                            print("After map filter.")
                            #! Making map 
                            map = folium.Map(location=[latitude_user, longitude_user], zoom_start=12)
                            folium.CircleMarker(
                                location=[latitude_user, longitude_user],
                                radius=80,
                                color='blue',
                                fill=True,
                                fill_color='blue',
                                fill_opacity=0.2
                            ).add_to(map)

                            marker_cluster = MarkerCluster().add_to(map)

                            route_points = []
                            for loc in data:
                                route_points.append([loc['latitude'], loc['longitude']])
                                iframe = IFrame(loc['info'], width=300, height=200)
                                popup = folium.Popup(iframe, max_width=800)
                                folium.Marker(
                                    location=[loc['latitude'], loc['longitude']],
                                    popup=popup,
                                    icon=folium.Icon(color='red')
                                ).add_to(marker_cluster)

                            if Morning:
                                for loc in morning_crime_data:
                                    iframe = IFrame(loc['info'], width=300, height=200)
                                    popup = folium.Popup(iframe, max_width=800)
                                    folium.Marker(
                                        location=[loc['latitude'], loc['longitude']],
                                        popup=popup,
                                        icon=folium.Icon(color='green', icon="flag")
                                    ).add_to(marker_cluster)

                            if Afternoon:
                                for loc in afternoon_crime_data:
                                    iframe = IFrame(loc['info'], width=300, height=200)
                                    popup = folium.Popup(iframe, max_width=800)
                                    folium.Marker(
                                        location=[loc['latitude'], loc['longitude']],
                                        popup=popup,
                                        icon=folium.Icon(color='green', icon="flag")
                                    ).add_to(marker_cluster)

                            if Evening:
                                for loc in evening_crime_data:
                                    iframe = IFrame(loc['info'], width=300, height=200)
                                    popup = folium.Popup(iframe, max_width=800)
                                    folium.Marker(
                                        location=[loc['latitude'], loc['longitude']],
                                        popup=popup,
                                        icon=folium.Icon(color='green', icon="flag")
                                    ).add_to(marker_cluster)

                            # if not Morning and not Afternoon and not Evening:
                            #     for loc in crime_data:
                            #     # the place to add additional data
                            #     ###print("loc:", loc)
                            #         iframe = IFrame(loc['info'], width=300, height=200)
                            #         popup = folium.Popup(iframe, max_width=800)
                            #         folium.Marker(
                            #             location=[loc['latitude'], loc['longitude']],
                            #             popup=popup,
                            #             icon=folium.Icon(color='green', icon="flag")
                            #         ).add_to(marker_cluster)

                            service_crime_map_title = f"{classified_service_type} Services & Crime near {zipcode}"
                            service_crime_map_header = GoogleTranslator(source='auto', target=input_language).translate(str(service_crime_map_title))
                            st.header(service_crime_map_header)
                            folium_static(map, width=800, height=600)  # Adjust width and height as needed


                            Options = [None]
                            for service in option_services: 
                                Options.append(str(service[0]))
                            select_service_title = GoogleTranslator(source='auto', target=input_language).translate(str('Select a service'))
                            selected_option = st.selectbox(select_service_title, Options)
                            if not selected_option is None:
                                send_email(input_language)

                        else:
                            st.sidebar.error(f"Error: Unable to retrieve location information for ZIP code {zipcode}")
                    else:
                        st.error("Service type is not recognized. Please try again with a different service type.")
                except Exception as e:
                    st.error(f"Error during classification or file handling: {e}")
            else:
                if not service_type:
                    st.error("Could not extract the type of service from your query. Please try rephrasing.")
                if not zipcode:
                    st.error("Could not extract the ZIP code from your query. Please try rephrasing.")
        else:
            st.write('Sorry, we cannot extract enough information to help you. You can refer to the following example: Find me a food pantry near market east && families in Philadelphia.')