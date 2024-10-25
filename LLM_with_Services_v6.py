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

# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, api_key, conversation_history):
    openai.api_key = api_key
    extraction_instruction = ("Extract the type of service and zipcode from the following user query. "
                              "If the user doesn't provide a zipcode and only provides a general landmark, "
                              "please try to provide the corresponding zipcode for that landmark. "
                              "Please ensure that if the input is in Spanish, translate the input to English first, the responses are provided in English."
                              "Only provide the service type and the zipcode.")
    combined_query = f"{extraction_instruction}\n{question}"
    full_conversation = conversation_history + [{"role": "user", "content": combined_query}]
    response = openai.ChatCompletion.create(model="gpt-4", messages=full_conversation)
    
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
        model="gpt-4",  # Updated to use the latest and more advanced model
        messages=[
            {"role": "system", "content": "You are a classifier that categorizes service types."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    st.write(response)
    raw_category = response['choices'][0]['message']['content'].strip() if response['choices'] else "Other"
    st.write("Raw Model Response for Classification:", raw_category)
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
    
    data = []
    for index, row in df.iterrows():
        # Merge and clean services information
        main_services = eval(row['Main_Services']) if pd.notna(row['Main_Services']) else []
        other_services = eval(row['Other_Services']) if pd.notna(row['Other_Services']) else []
        services = main_services + other_services
        services = [service for service in services if service != 'None']
        
        # Get the opening hours for the current day
        opening_hours_today = row[current_day] if pd.notna(row[current_day]) else 'Unavailable'
        #current_time = now.strftime("%H:%M")
        #print("opening_hours_today: ", opening_hours_today, opening_hours_today[:7])

        # determine whether the service is open or closed
        #timeStart = datetime.strptime(opening_hours_today[:7], "%I:%M%p")
        ###timeEnd = datetime.strptime(timeEnd, "%I:%M%p")
        #timeNow = datetime.strptime(current_time, "%I:%M%p")



        # Format popup information to include today's hours
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

    return data

# read crime data
@st.cache_data
def read_crime_data(df):
    data = []
    for index, row in df.iterrows():
        # Format popup information to include today's hours
        info = f"""
            <strong>Crime Type:</strong> {row['text_general_code']}<br>
            <strong>Time:</strong> {row['dispatch_date_time']}<br>
            <strong>Geolocation:</strong> {row['location_block']}
        """

        # Append this information along with latitude and longitude
        data.append({
            'latitude': row['lat'],
            'longitude': row['lng'],
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

# Streamlit UI
img = Image.open('dream_kg_logo_v2.png')
st.image(img)
st.markdown("# User Input")
st.markdown("### Ask me about available services:")
user_query = st.text_input("Enter your query (e.g., 'I need mental health support near 19122')", key="user_query")
# add picture option
###img_file_buffer = st.camera_input("Take a picture") # "Take a picture"
# Submit button
submit_button = st.button("Submit")


# Initialize global variables
conversation_history = []
api_key = 'sk-None-F4ump4p48TwdpLJ8zPDLT3BlbkFJRbUZiveN9s2NIBKlO3GF'
#'sk-proj-yu46NPL9HS19sGFQApnWT3BlbkFJNDYsqs3gsWPJFpvmGwsZ'
#'sk-proj-fTGwpbf19mQlPAx6iLRCT3BlbkFJMfQ8QAelwblEgiVmOl0p'  # Replace this with your actual OpenAI API key


def parse_extracted_info(extracted_info):
    # Using regular expressions to find service type and zipcode robustly
    service_match = re.search(r"service(?: type)?:\s*(.+)", extracted_info, re.I)
    zipcode_match = re.search(r"zipcode:\s*(\d+)", extracted_info, re.I)
    
    service_type = service_match.group(1).title() if service_match else ""
    zipcode = zipcode_match.group(1) if zipcode_match else ""
    
    return service_type, zipcode
    
if submit_button:
    #print("user_query:", user_query)
    response = ask_openai_for_service_extraction(user_query, api_key, conversation_history)
    print("user_query:", user_query)
    input_language = detect(user_query)
    if input_language == 'en':
        if response.choices:
            extracted_info = response.choices[0].message['content'].strip()
            st.write("Extracted Information:", extracted_info)

            service_type, zipcode = parse_extracted_info(extracted_info)
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

            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")

            if service_type and zipcode:
                try:
                    classified_service_type = classify_service_type(service_type, api_key)
                    st.write("Current Time in Eastern Standard Time: ", current_time)
                    st.write("Type of Service:", classified_service_type)
                    st.write("Zipcode:", zipcode)
                    st.write("In Year 2024, the Number of Crime Incidents in Zipcode " + zipcode + " is ",
                             str(number_of_crimes), ",", " where there are ", str(number_of_property_crimes), "property victimization and there are ", str(number_of_personal_crimes), "personal victimization")
                    st.write("In Year 2024, the Frequencies of Different Crime Incidents in Zipcode are as follows " + zipcode + ":",
                        str(crime_frequency))

                    # To read image file buffer as a PIL Image:
                    #img = Image.open(img_file_buffer)
                    #where_location = identify_image_geolocation(img, api_key)
                    #st.write("where_location:", where_location)

                    # To convert PIL Image to numpy array:
                    #img_array = np.array(img)
                    #print("img_array: ", img_array)
                    #st.write("Img_array:", img_array)

                    if classified_service_type != "Other":
                        service_files = {
                            "Shelter": "Final_Temporary_Shelter_20240723.csv",
                            "Mental Health": "Final_Mental_Health_20240723.csv",
                            "Food": "Final_Emergency_Food_20240723.csv"
                        }
                        datafile = service_files[classified_service_type]
                        df = pd.read_csv(datafile)
                        data = read_data(df)

                        # load crime data (till 07/2024) for visualization
                        crime_data_df = pd.read_csv('recent_philly_incidents_2024.csv')
                        crime_data = read_crime_data(crime_data_df)

                        # Use pgeocode for geocoding
                        nomi = pgeocode.Nominatim('us')
                        location_info = nomi.query_postal_code(zipcode)

                        if not location_info.empty:
                            latitude_user = location_info['latitude']
                            longitude_user = location_info['longitude']
                            city_name = location_info['place_name']
                            client = Steamship(workspace="gpt-4-g4d")

                            # Create an instance of this generator
                            generator = client.use_plugin('gpt-4', config={"temperature": 0.7, "n": 5})
                            geolocation_query = "Just list the their names with comma. Please find only five famous buildings or benchmarks close to the location: latitidue: " + str(
                                latitude_user) + ", " + "longitude: " + str(longitude_user)
                            task = generator.generate(text=geolocation_query)
                            task.wait()
                            message = task.output.blocks
                            message = [i.text.strip() for i in message]
                            st.write(f"Coordinates for {zipcode} ({city_name}): {latitude_user}, {longitude_user}")
                            st.write(f"Architectural Buildings Around: {message[0]}")
                            #translate_client = Steamship(workspace="gpt-4-g4d")
                            #translate_generator = translate_client.use_plugin('gpt-4',config={"temperature": 0.7, "n": 5})
                            #translate_task = translate_generator.generate(text="Translate the answer to Spanish: " + message[0])
                            #translate_task.wait()
                            #translate_message = translate_task.output.blocks
                            #translate_message = [i.text.strip() for i in translate_message]
                            #print("translate_message:", translate_message)

                            map = folium.Map(location=[latitude_user, longitude_user], zoom_start=12)
                            folium.CircleMarker(
                                location=[latitude_user, longitude_user],
                                radius=80,
                                color='blue',
                                fill=True,
                                fill_color='blue',
                                fill_opacity=0.2
                            ).add_to(map)
                            '''
                            # reference from https://www.ostirion.net/post/mapping-geo-data-with-folium
                            # using satellite map
                            # 可加可不加
                            tile = folium.TileLayer(
                                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                                attr='Esri',
                                name='Esri Satellite',
                                overlay=False,
                                control=True
                            ).add_to(map)
                            # done
                            '''
                            marker_cluster = MarkerCluster().add_to(map)

                            for loc in data:
                                # the place to add additional data
                                ###print("loc:", loc)
                                iframe = IFrame(loc['info'], width=300, height=200)
                                popup = folium.Popup(iframe, max_width=800)
                                folium.Marker(
                                    location=[loc['latitude'], loc['longitude']],
                                    popup=popup,
                                    icon=folium.Icon(color='red')
                                ).add_to(marker_cluster)
                            # adding crime information into map will take a long time
                            for loc in crime_data:
                                # the place to add additional data
                                ###print("loc:", loc)
                                iframe = IFrame(loc['info'], width=300, height=200)
                                popup = folium.Popup(iframe, max_width=800)
                                folium.Marker(
                                    location=[loc['latitude'], loc['longitude']],
                                    popup=popup,
                                    icon=folium.Icon(color='green', icon="flag")
                                ).add_to(marker_cluster)

                            st.header(f"{classified_service_type} Services near {zipcode}")
                            folium_static(map, width=800, height=600)  # Adjust width and height as needed

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
    elif input_language == 'es':
        if response.choices:
            extracted_info = response.choices[0].message['content'].strip()
            st.write("Información extraída:", extracted_info)

            service_type, zipcode = parse_extracted_info(extracted_info)
            crime_data_df = pd.read_csv('Final_Philadelphia_Crime_Data_2023.csv')
            crime_data = crime_data_df.values
            number_of_crimes = np.where(int(zipcode) == crime_data[:, 18])[0].shape[0]
            crime_frequency = crime_data_df.loc[
                np.where(int(zipcode) == crime_data[:, 18])[0], 'text_general_code'].value_counts().to_dict()

            now = datetime.now()

            current_time = now.strftime("%H:%M:%S")

            if service_type and zipcode:
                try:
                    classified_service_type = classify_service_type(service_type, api_key)
                    st.write("Hora actual en hora estándar del este: ", current_time)
                    st.write("Tipo de servicio:", classified_service_type)
                    st.write("Código postal:", zipcode)
                    st.write("En el año 2023, el número de incidentes delictivos en el código postal " + zipcode + ":",
                             str(number_of_crimes))
                    translate_crime_frequency_client = Steamship(workspace="gpt-4-g4d")
                    translate_crime_frequency_generator = translate_crime_frequency_client.use_plugin('gpt-4',
                                                                      config={"temperature": 0.7, "n": 5})
                    translate_crime_frequency_task = translate_crime_frequency_generator.generate(
                        text="Translate the answer to Spanish: " + str(crime_frequency))
                    translate_crime_frequency_task.wait()
                    translate_crime_frequency_message = translate_crime_frequency_task.output.blocks
                    translate_crime_frequency_message = [i.text.strip() for i in translate_crime_frequency_message]
                    st.write(
                        "En el año 2023, las frecuencias de diferentes incidentes delictivos en el código postal son las siguientes " + zipcode + ":",
                        translate_crime_frequency_message[1] + "}")

                    if classified_service_type != "Other":
                        service_files = {
                            "Shelter": "Final_Temporary_Shelter_20240423.csv",
                            "Mental Health": "Final_Mental_Health_20240423.csv",
                            "Food": "Final_Emergency_Food_20240423.csv"
                        }
                        datafile = service_files[classified_service_type]
                        df = pd.read_csv(datafile)
                        data = translated_read_data(df)

                        # Use pgeocode for geocoding
                        nomi = pgeocode.Nominatim('us')
                        location_info = nomi.query_postal_code(zipcode)

                        if not location_info.empty:
                            latitude_user = location_info['latitude']
                            longitude_user = location_info['longitude']
                            city_name = location_info['place_name']
                            client = Steamship(workspace="gpt-4-g4d")

                            # Create an instance of this generator
                            generator = client.use_plugin('gpt-4', config={"temperature": 0.7, "n": 5})
                            geolocation_query = "Just list the their names with comma. Please find only five famous buildings or benchmarks close to the location: latitidue: " + str(
                                latitude_user) + ", " + "longitude: " + str(longitude_user)
                            task = generator.generate(text=geolocation_query)
                            task.wait()
                            message = task.output.blocks
                            message = [i.text.strip() for i in message]
                            st.write(f"Coordenadas para {zipcode} ({city_name}): {latitude_user}, {longitude_user}")
                            #st.write(f"Edificios arquitectónicos alrededor: {message[0]}")
                            translate_client = Steamship(workspace="gpt-4-g4d")
                            translate_generator = translate_client.use_plugin('gpt-4',
                                                                              config={"temperature": 0.7, "n": 5})
                            translate_task = translate_generator.generate(
                                text="Translate the answer to Spanish: " + message[0])
                            translate_task.wait()
                            translate_message = translate_task.output.blocks
                            translate_message = [i.text.strip() for i in translate_message]
                            #print("translate_message:", translate_message)
                            st.write(
                                "Edificios arquitectónicos alrededor:",
                                translate_message[0])
                            #st.write(
                            #    "En el año 2023, las frecuencias de diferentes incidentes delictivos en el código postal son las siguientes " + zipcode + ":",
                            #    translate_message[0])

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

                            for loc in data:
                                iframe = IFrame(loc['info'], width=300, height=200)
                                popup = folium.Popup(iframe, max_width=800)
                                folium.Marker(
                                    location=[loc['latitude'], loc['longitude']],
                                    popup=popup,
                                    icon=folium.Icon(color='red')
                                ).add_to(marker_cluster)

                            st.header(f"{classified_service_type} Services near {zipcode}")
                            folium_static(map, width=800, height=600)  # Adjust width and height as needed

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


