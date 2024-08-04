from flask import Flask, request, render_template
import joblib
import numpy as np
import pickle


app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from form data
    try:
        # Get the input data from the form
        features = [float(request.form[f'feature{i}']) for i in range(1, 42)]  # Adjust range for your 41 features
        final_features = [np.array(features)]
    
    # Predict using the model
        prediction = model.predict(final_features)
        print(prediction)
    
    # Render prediction result
        return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction[0]))

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
