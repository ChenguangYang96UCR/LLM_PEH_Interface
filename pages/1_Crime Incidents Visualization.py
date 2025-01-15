import branca
import folium
import folium.plugins
import streamlit as st
import pandas as pd
from streamlit_folium import st_folium
from pathlib import Path
from branca.element import IFrame
from folium.plugins import MarkerCluster
from datetime import datetime


def read_data(df):
    df = df.dropna(subset=['Latitude', 'Longitude'])  # Ensure we have the necessary location data
    # Get the current day and time
    now = datetime.now()
    current_day = now.strftime('%A')  # e.g., 'Monday'

    data = []
    for index, row in df.iterrows():
        main_services = eval(row['Main_Services']) if pd.notna(row['Main_Services']) else []
        other_services = eval(row['Other_Services']) if pd.notna(row['Other_Services']) else []
        services = main_services + other_services
        services = [service for service in services if service != 'None']

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

        data.append({
            'latitude': row['Latitude'],
            'longitude': row['Longitude'],
            'info': info
        })

    return data


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

st.set_page_config(
    layout="wide",
    page_title="streamlit-folium documentation: Misc Examples",
    page_icon=":pencil:",
)


### 10/24 - edited
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background-color: #fcb290;
    }
</style>
""", unsafe_allow_html=True)

#page = st.radio("Select map type", ["Branca figure"], index=0)

page = st.radio("Select Visualization Type", ["Crime Incidents Visualization", "Food Pantry Visualization", "Mental Health Visualization", "Temporary Shelter Visualization"], index=0)


map = folium.Map(location=[39.949610, -75.150282], zoom_start=12)

if page == "Crime Incidents Visualization":
    """
    # 🚔 Crime Incidents January 9, 2025 - Present
    """
    marker_cluster = MarkerCluster().add_to(map)
    p = Path(__file__).parent / "philly_crime_incident_util_01112025.csv"
    crime_data_df = pd.read_csv(p)
    crime_data = read_crime_data(crime_data_df)

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
elif page == "Mental Health Visualization":
    """
    # 🩺 Mental Health
    """
    marker_cluster = MarkerCluster().add_to(map)
    p = Path(__file__).parent / "Final_Mental_Health_20240723.csv"
    df = pd.read_csv(p)
    data = read_data(df)
    route_points = []
    for loc in data:
        # the place to add additional data
        print("loc:", loc)
        route_points.append([loc['latitude'], loc['longitude']])
        iframe = IFrame(loc['info'], width=300, height=200)
        popup = folium.Popup(iframe, max_width=800)
        folium.Marker(
            location=[loc['latitude'], loc['longitude']],
            popup=popup,
            icon=folium.Icon(color='red')
        ).add_to(marker_cluster)

elif page == "Temporary Shelter Visualization":
    """
    # 🏠 Temporary Shelter
    """
    marker_cluster = MarkerCluster().add_to(map)
    p = Path(__file__).parent / "Final_Temporary_Shelter_20240723.csv"
    df = pd.read_csv(p)
    data = read_data(df)
    route_points = []
    for loc in data:
        # the place to add additional data
        print("loc:", loc)
        route_points.append([loc['latitude'], loc['longitude']])
        iframe = IFrame(loc['info'], width=300, height=200)
        popup = folium.Popup(iframe, max_width=800)
        folium.Marker(
            location=[loc['latitude'], loc['longitude']],
            popup=popup,
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)

elif page == "Food Pantry Visualization":
    """
    # 🥖 Food Pantry
    """
    marker_cluster = MarkerCluster().add_to(map)
    p = Path(__file__).parent / "Final_Emergency_Food_20240723.csv"
    df = pd.read_csv(p)
    data = read_data(df)
    route_points = []
    for loc in data:
        # the place to add additional data
        print("loc:", loc)
        route_points.append([loc['latitude'], loc['longitude']])
        iframe = IFrame(loc['info'], width=300, height=200)
        popup = folium.Popup(iframe, max_width=800)
        folium.Marker(
            location=[loc['latitude'], loc['longitude']],
            popup=popup,
            icon=folium.Icon(color='black')
        ).add_to(marker_cluster)




st_folium(map, width=2000, height=500)


#df = pd.read_csv(datafile)
#print("df:", df)
#data = read_data(df)





#with st.echo():
    # call to render Folium map in Streamlit
    #st_folium(map, width=2000, height=500, returned_objects=[], debug=True)
    #st_folium(map, width=2000, height=500)