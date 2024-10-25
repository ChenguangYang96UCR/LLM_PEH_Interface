import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime
import pgeocode
import pandas as pd
import openai
from branca.element import IFrame

# Function to query OpenAI API for extracting the service and zipcode
def ask_openai_for_service_extraction(question, api_key, conversation_history):
    openai.api_key = api_key
    extraction_instruction = "Extract the type of service and zipcode from the following user query:"
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
    response = openai.Completion.create(engine="davinci-002", prompt=prompt, max_tokens=60)
    
    raw_category = response.choices[0].text.strip() if response.choices else "Other"
    st.write("Raw Model Response for Classification:", raw_category)

    raw_category = raw_category.split('\n')[0]
    raw_category = raw_category.replace('1-', '').replace('2-', '').replace('3-', '').strip()

    for standard_category in ['Food', 'Shelter', 'Mental Health']:
        if standard_category.lower() in raw_category.lower():
            return standard_category

    return "Other"

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

        # Format popup information to include today's hours
        info = f"""
            <strong>{row['Service_Name']}</strong><br>
            <strong>Today's Hours:</strong> {opening_hours_today}<br>
            <strong>Services:</strong> {', '.join(services)}<br>
            <strong>Serving:</strong> {row['Serving']}<br>
            <strong>Phone Number:</strong> {row['Phone_Number']}<br>
            <strong>Eligibility:</strong> {row['Eligibility']}<br>
            <strong>Languages:</strong> {row['Languages']}<br>
            <strong>Cost:</strong> {row['Cost']}
        """

        # Append this information along with latitude and longitude
        data.append({
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'info': info
        })

    return data

# Streamlit UI
st.markdown("# User Input")
st.markdown("### Ask me about available services:")
user_query = st.text_input("Enter your query (e.g., 'I need mental health support near 19122')", key="user_query")

# Submit button
submit_button = st.button("Submit")

# Initialize global variables
conversation_history = []
api_key = 'sk-proj-yu46NPL9HS19sGFQApnWT3BlbkFJNDYsqs3gsWPJFpvmGwsZ' #'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'  # Replace this with your actual OpenAI API key

if submit_button:
    response = ask_openai_for_service_extraction(user_query, api_key, conversation_history)
    if response.choices:
        extracted_info = response.choices[0].message['content'].strip()
        
        # Debugging: Display extracted information
        st.write("Extracted Information:", extracted_info)
        
        lines = extracted_info.lower().split('\n')
        parsed_info = {}
        for line in lines:
            line = (line.replace('type of service:', 'service type:')
                        .replace('service:', 'service type:')
                        .replace('zipcode:', 'zip code:'))
            parts = line.split(':', 1)
            if len(parts) == 2:
                key, value = parts
                parsed_info[key.strip()] = value.strip()

        raw_service_type = parsed_info.get("service type", "").title()
        zipcode = parsed_info.get("zip code", "")
        
        if raw_service_type and zipcode:
            classified_service_type = classify_service_type(raw_service_type, api_key)
            st.write("Type of Service:", classified_service_type)
            st.write("Zipcode:", zipcode)
            
            if classified_service_type != "Other":
                service_files = {
                    "Shelter": "Final_Temporary_Shelter_20240109.csv",
                    "Mental Health": "Final_Mental_Health_20240109.csv",
                    "Food": "Final_Emergency_Food_20240109.csv"
                }
                datafile = service_files[classified_service_type]
                df = pd.read_csv(datafile)
                data = read_data(df)
                
                # Use pgeocode for geocoding
                nomi = pgeocode.Nominatim('us')
                location_info = nomi.query_postal_code(zipcode)

                if not location_info.empty:
                    latitude_user = location_info['latitude']
                    longitude_user = location_info['longitude']
                    city_name = location_info['place_name']
                    st.write(f"Coordinates for {zipcode} ({city_name}): {latitude_user}, {longitude_user}")

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
                        popup = folium.Popup(iframe, max_width=500)
                        folium.Marker(
                            location=[loc['latitude'], loc['longitude']],
                            popup=popup,
                            icon=folium.Icon(color='red')
                        ).add_to(marker_cluster)
                    
                    st.header(f"{classified_service_type} Services near {zipcode}")
                    #folium_static(map)
                    folium_static(map, width=800, height=600)  # Adjust width and height as needed

                else:
                    st.sidebar.error(f"Error: Unable to retrieve location information for ZIP code {zipcode}")
            else:
                st.error("Service type is not recognized. Please try again with a different service type.")
        else:
            if not raw_service_type:
                st.error("Could not extract the type of service from your query. Please try rephrasing.")
            if not zipcode:
                st.error("Could not extract the ZIP code from your query. Please try rephrasing.")

