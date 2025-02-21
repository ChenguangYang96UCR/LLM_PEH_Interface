import geopy
import pandas as pd


def get_zipcode(df, geolocator, lat_field, lon_field):
    location = geolocator.reverse((df[lat_field], df[lon_field]))
    return location.raw['address']['postcode']


geolocator = geopy.Nominatim(user_agent='1234')
lat_array = []
lon_array = []

cvs_path = './three_days_philly_incidents_2025.csv'
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
df.to_csv('three_days_philly_incidents_with_zipcode_2025.csv', index=False)