from flask import Flask, request, jsonify, render_template_string
import joblib
import os
import re

app = Flask(__name__)

# Load your trained model
model_path = 'C:\\Users\\MITS\\Downloads\\Spam Email Detection\\Spam Email Detection\\model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load your vectorizer
vectorizer_path = 'C:\\Users\\MITS\\Downloads\\Spam Email Detection\\Spam Email Detection\\vectorizer.pkl'
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
else:
    raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

def preprocess_text(text):
    # Remove unwanted characters and preprocess the text
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove all non-word characters
    text = text.strip()  # Remove leading and trailing whitespace
    return text

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Get the input data from the request
            data = request.form.get('data')
            if data is None:
                return jsonify({'error': 'No data provided'}), 400

            # Preprocess the input data
            data = preprocess_text(data)

            # Convert the input data to a format suitable for prediction
            data_vectorized = vectorizer.transform([data])

            # Make predictions using your model
            predictions = model.predict(data_vectorized)

            # Map the prediction to "Spam" or "Not Spam"
            prediction_label = "Spam" if predictions[0] == 1 else "Not Spam"

            # Return the prediction as a JSON response
            return jsonify({'prediction': prediction_label})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # Render a simple HTML form for input
        return render_template_string('''
            <!doctype html>
<html>
<head>
    <title>Spam Email Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }
        .container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            width: 80%;
            max-width: 600px;
            text-align: center;
        }
        h1 {
            color: #333333;
        }
        textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #cccccc;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        input[type="submit"] {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Enter text to predict if it's spam</h1>
        <form method="post">
            <textarea name="data" rows="4" cols="50" placeholder="Enter your email text here..."></textarea><br>
            <input type="submit" value="Predict">
        </form>
    </div>
</body>
</html>

        ''')

@app.errorhandler(404)
def not_found(error):
    return '404 Not Found', 404

if __name__ == '__main__':
    app.run(debug=True)