import pandas as pd
import numpy as np
#import geopy

'''
philly_crime_data = pd.read_csv('incidents_part1_part2.csv')
print(philly_crime_data.shape)
#print(philly_crime_data)
#print(philly_crime_data[:125].isnull())
print(philly_crime_data['lat'].isnull().sum())
philly_crime_data = philly_crime_data.dropna(subset=['lat'])
#philly_crime_data.dropna(subset=['lat'], inplace=True)
print(philly_crime_data.shape)
print(philly_crime_data['lat'].isnull().sum())
philly_crime_data.to_csv('final_incidents_part1_part2.csv')
'''

crime_data_df = pd.read_csv('Final_Philadelphia_Crime_Data_2023.csv')
crime_data = crime_data_df.values
zipcode = 19122
number_of_crimes_index = np.where(int(zipcode) == crime_data[:, 18])[0]
sub_crime_data = crime_data[number_of_crimes_index,]
number_of_personal_crimes_index = np.where(('Other Assaults' == sub_crime_data[:, 13]) | ('Robbery No Firearm' == sub_crime_data[:, 13])
										   | ('Robbery Firearm' == sub_crime_data[:, 13]) | ('Offenses Against Family and Children' == sub_crime_data[:, 13])
										   | ('Other Sex Offenses (Not Commercialized)' == sub_crime_data[:, 13]))[0]
print(number_of_personal_crimes_index.shape)



from datetime import datetime

def isNowInTimePeriod(startTime, endTime, nowTime):
    if startTime < endTime:
        return nowTime >= startTime and nowTime <= endTime
    else: #Over midnight
        return nowTime >= startTime or nowTime <= endTime


timeStart = '8:00AM'
timeEnd = '11:00PM'
timeNow = '12:59AM'
timeStart = datetime.strptime(timeStart, "%I:%M%p")
timeEnd = datetime.strptime(timeEnd, "%I:%M%p")
timeNow = datetime.strptime(timeNow, "%I:%M%p")
print(timeNow)

print(isNowInTimePeriod(timeStart, timeEnd, timeNow))