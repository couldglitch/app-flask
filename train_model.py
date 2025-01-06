import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load your dataset
data = pd.read_csv('bus_crowd_data_one_year.csv')

# Preprocess the data
# Convert categorical variables to numerical
data['day'] = data['day'].astype('category').cat.codes
data['weather'] = data['weather'].astype('category').cat.codes
data['special_event'] = data['special_event'].astype('category').cat.codes

# Features and target variable
X = data[['time', 'day', 'weather', 'special_event']]
y = data['crowd_level']

# Convert time to a numerical format (e.g., total minutes from midnight)
X['time'] = pd.to_datetime(X['time'], format='%H:%M').dt.hour * 60 + pd.to_datetime(X['time'], format='%H:%M').dt.minute

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model to a file
pickle.dump(model, open('bus_crowd_model.pkl', 'wb'))

print("Model trained and saved successfully!")