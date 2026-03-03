from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Load the Pre-trained Models
try:
    print("Loading ML model...")
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    print("Model loaded successfully!")

except Exception as e:
    print(f"Error loading model: {e}")
    vectorizer = None
    model = None


# Home Route (Loads Frontend)
@app.route('/')
def home():
    return render_template('index.html')


# Prediction API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        message = data.get('message')

        if not message:
            return jsonify({'error': 'No message provided'}), 400

        # Transform input text
        processed_message = vectorizer.transform([message])

        # Predict
        prediction = model.predict(processed_message)
        is_spam = bool(prediction[0])

        # Return result
        result = 'spam' if is_spam else 'ham'
        return jsonify({'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Run the Server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
