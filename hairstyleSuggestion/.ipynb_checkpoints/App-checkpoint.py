from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the trained model in app context
model = joblib.load('hairstyle_model.pkl')  # Ensure this path is correct

# Helper function to make predictions
def make_prediction(data):
    try:
        # Extract features from the request
        face_shape = int(data.get('FaceShape'))  # Convert FaceShape to integer
        gender = 1 if data.get('Gender', '').lower() == 'male' else 0  # Convert Gender to binary
        age_group = int(data.get('AgeGroup'))  # Convert AgeGroup to integer

        # Prepare the feature vector for prediction
        features = np.array([[face_shape, gender, age_group]])

        # Make a prediction
        prediction = model.predict(features)

        # Return the result
        result = {'RecommendedHairstyle': str(prediction[0])}  # Adjust based on your model's output
        return result, 200

    except KeyError as ke:
        return {'error': f'Missing key: {str(ke)}'}, 400
    except ValueError as ve:
        return {'error': f'Invalid value: {str(ve)}'}, 400
    except Exception as e:
        return {'error': str(e)}, 500

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("Received data:", data)  # Log received data for debugging
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    result, status_code = make_prediction(data)
    return jsonify(result), status_code

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)