from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and TF-IDF vectorizer
model = joblib.load('fake_job_model.pkl')  # Your saved model
tfidf = joblib.load('tfidf_vectorizer.pkl')  # Your saved TF-IDF vectorizer

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    title = request.form.get('title')
    company_profile = request.form.get('company_profile')
    description = request.form.get('description')
    requirements = request.form.get('requirements')
    benefits = request.form.get('benefits')

    # Preprocess and vectorize the input
    vectorized_text = tfidf.transform([title, company_profile, description, requirements, benefits])

    # Predict using the loaded model
    prediction = model.predict(vectorized_text)
    
    # Return result
    result = {'prediction': 'Fake' if prediction[0] == 1 else 'Real'}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
