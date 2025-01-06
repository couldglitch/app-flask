import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)
time_slots = [f"{str(hour).zfill(2)}:00" for hour in range(24)]  # 24-hour format
weather_conditions = ['Sunny', 'Rainy', 'Cloudy']
special_events = ['Holiday', 'Concert', 'None']  # Three options for each day

# Generate date range
date_range = pd.date_range(start=start_date, end=end_date)

# Create an empty list to hold the data
data = []

# Generate data for each day in the date range
for single_date in date_range:
    # Randomly choose whether the day is a Holiday, Concert, or None
    event_type = np.random.choice(special_events)

    for time in time_slots:
        # Randomly select weather
        weather = np.random.choice(weather_conditions)

        # Determine crowd level based on conditions (this is a simple heuristic)
        if weather == 'Sunny':
            crowd_level = np.random.randint(50, 100)
        elif weather == 'Rainy':
            crowd_level = np.random.randint(20, 60)
        else:  # Cloudy
            crowd_level = np.random.randint(30, 70)

        # Adjust crowd level for the selected special event
        if event_type == 'Holiday':
            crowd_level += np.random.randint(10, 30)  # Increase crowd level for holidays
        elif event_type == 'Concert':
            crowd_level += np.random.randint(20, 50)  # Increase crowd level for concerts
        # No adjustment needed for 'None'

        # Append the data
        data.append({
            'date': single_date.strftime('%Y-%m-%d'),  # Format date as YYYY-MM-DD
            'time': time,
            'day': single_date.strftime('%A'),  # Get the day of the week
            'weather': weather,
            'special_event': event_type,  # Same event for the whole day
            'crowd_level': crowd_level
        })

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('bus_crowd_data_one_year.csv', index=False)

print("Data generated and saved to 'bus_crowd_data_one_year.csv'")