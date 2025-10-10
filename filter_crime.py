import pandas as pd
from datetime import datetime
import os
from datetime import datetime
import geopy

def get_zipcode(df, geolocator, lat_field, lon_field):
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location.raw['address']['postcode']


def extract_dispatch_time(time_string):
    time_part = time_string.split(" ")[1]  # "17:58:00"
    hour = time_part.split(":")[0]
    return int(hour)

def sperate_crime_according_time(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        morning_filtered_df = df.iloc[0:0].copy()
        afternoon_filtered_df = df.iloc[0:0].copy()
        evening_filtered_df = df.loc[0:0].copy()
        Index = 0
        for crime in df['dispatch_date_time']:
            hour = extract_dispatch_time(crime)
            # Morning filter
            if 6 <= hour <= 12:
                morning_filtered_df.loc[len(morning_filtered_df)] = df.loc[Index]
            # Afternoon filter
            if 12 < hour <= 18:
                afternoon_filtered_df.loc[len(afternoon_filtered_df)] = df.loc[Index]
            # Evening filter
            if 18 < hour <= 24 or 0 <= hour < 6:
                evening_filtered_df.loc[len(evening_filtered_df)] = df.loc[Index]
            Index = Index + 1

        morning_filtered_df.to_csv("./files/three_days_philly_incidents_2025_morning.csv", index=False)
        afternoon_filtered_df.to_csv("./files/three_days_philly_incidents_2025_afternoon.csv", index=False)
        evening_filtered_df.to_csv("./files/three_days_philly_incidents_2025_evening.csv", index=False)

if __name__ == '__main__':
    # 0. Add every indicident's ucr code based on their crime type
    cvs_path = './files/incidents_part1_part2-2.csv'
    file = pd.read_csv(cvs_path)
    crime_ucr_map = {"Thefts" : "23D", 'Theft from Vehicle' : '23F', 'Motor Vehicle Theft' : '240', 'Aggravated Assault Firearm' : '13A', 
                     'Aggravated Assault No Firearm' : '13A', 'Other Assaults' : '13B', 'Robbery Firearm' : '120', 'Robbery No Firearm' : '120',
                     'Burglary Non-Residential' : '220', 'Burglary Residential' : '220', 'Rape' : '11A', 'Homicide - Criminal' : '09A', 'Vandalism/Criminal Mischief' : '290',
                     'Fraud' : '26B', 'All Other Offenses' : '90Z', 'Other Sex Offenses (Not Commercialized)': '11C', 'DRIVING UNDER THE INFLUENCE' : '90D', 
                     'Receiving Stolen Property' : '280', 'Embezzlement' : '270', 'Weapon Violations' : '520', 'Arson' : '200', 'Disorderly Conduct' : '90C',
                     'Prostitution and Commercialized Vice' : '40A', 'Offenses Against Family and Children' : '90F', 'Forgery and Counterfeiting' : '250', 'Public Drunkenness' : '90E',
                     'Narcotic / Drug Law Violations' : '35A'}
    
    ucr_code_list = []
    for crime_type in file['text_general_code']:
        ucr_code = crime_ucr_map[crime_type]
        ucr_code_list.append(ucr_code)

    file['ucr_code'] = ucr_code_list
    file.to_csv(cvs_path, index=False)

    # 1. Extract three days philadelphia indcidents
    df = pd.read_csv(cvs_path)
    filtered_df = df.iloc[0:0].copy()
    Index = 0
    for crime in df['dispatch_date']:
        print(crime)
        date_time = datetime.strptime(crime, "%Y-%m-%d")
        if date_time.day > 9 and date_time.month == 1:
            filtered_df.loc[len(filtered_df)] = df.loc[Index]
            # filtered_df = pd.concat([filtered_df, df.loc[Index]], ignore_index=True)
        Index = Index + 1

    filtered_df.to_csv("./files/three_days_philly_incidents_2025.csv", index=False)

    # 2. Get every incident's zipcode based on their geological information
    geolocator = geopy.Nominatim(user_agent='1234')
    lat_array = []
    lon_array = []

    cvs_path = './files/three_days_philly_incidents_2025.csv'
    df = pd.read_csv(cvs_path)

    for crime in df['lat']:
        print(crime)
        lat_array.append(float(crime))

    for crime in df['lng']:
        lon_array.append(float(crime))

    geo_df = pd.DataFrame({
        'Lat': lat_array,
        'Lon': lon_array
    })
    zipcodes = geo_df.apply(get_zipcode, axis=1, geolocator=geolocator, lat_field='Lat', lon_field='Lon')
    print(zipcodes)
    df['zipcode'] = zipcodes
    df.to_csv('./files/three_days_philly_incidents_with_zipcode_2025.csv', index=False)

    # 3. Seperate incidents to three time terms
    sperate_crime_according_time("./files/three_days_philly_incidents_with_zipcode_2025.csv")