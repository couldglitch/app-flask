from flask import Flask, request, jsonify, render_template, abort
import pickle
import pandas as pd
import logging

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load the trained model
try:
    with open('bus_crowd_model.pkl', 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    app.logger.error(f'Error loading model: {str(e)}')
    raise RuntimeError("Model could not be loaded.")

# Define the overcrowding threshold
OVERCROWDING_THRESHOLD = 50  # Example threshold

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Validate input data
    if not data or 'time' not in data or 'day' not in data or 'weather' not in data or 'special_event' not in data:
        abort(400, description="Invalid input data")

    try:
        # Process input data
        time = pd.to_datetime(data['time'], format='%H:%M').hour * 60 + pd.to_datetime(data['time'], format='%H:%M').minute
        day = int(data['day'])
        weather = int(data['weather'])
        special_event = int(data['special_event'])

        # Validate ranges
        if not (0 <= day <= 6) or not (0 <= weather <= 2) or not (0 <= special_event <= 2):
            abort(400, description="Invalid input values")

        # Prepare input data for prediction
        input_data = [[time, day, weather, special_event]]

        # Make prediction
        prediction = model.predict(input_data)

        # Check for overcrowding
        is_overcrowded = prediction[0] > OVERCROWDING_THRESHOLD

        # Ensure the response is serializable
        return jsonify({
            'crowd_level': float(prediction[0]),  # Convert to float for JSON serialization
            'overcrowded': bool(is_overcrowded)    # Ensure it's a boolean
        })

    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_query', methods=['GET'])
def predict_query():
    # Extract query parameters
    try:
        time = request.args.get('time')
        day = int(request.args.get('day'))
        weather = int(request.args.get('weather'))
        special_event = int(request.args.get('special_event'))

        # Validate ranges
        if not (0 <= day <= 6) or not (0 <= weather <= 2) or not (0 <= special_event <= 2):
            abort(400, description="Invalid input values")

        # Process input data
        time_in_minutes = pd.to_datetime(time, format='%H:%M').hour * 60 + pd.to_datetime(time, format='%H:%M').minute

        # Prepare input data for prediction
        input_data = [[time_in_minutes, day, weather, special_event]]

        # Make prediction
        prediction = model.predict(input_data)

        # Check for overcrowding
        is_overcrowded = prediction[0] > OVERCROWDING_THRESHOLD

        # Return the prediction as a JSON response
        return jsonify([{
            'crowd_level': float(prediction[0]),  # Convert to float for JSON serialization
            'overcrowded': bool(is_overcrowded)    # Ensure it's a boolean
        }])

    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad Request: " + str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource Not Found: " + str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error: " + str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
