import csv
from io import StringIO
from flask import Flask, render_template, request, redirect, url_for, make_response
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Store prediction history
history = []

@app.route('/')
def index():
    # Stats count
    spam_count = sum(1 for h in history if h['prediction'] == 'Spam')
    ham_count = sum(1 for h in history if h['prediction'] == 'Not Spam')
    total = len(history)

    # Category breakdown
    categories = {'Phishing': 0, 'Promotion': 0, 'General': 0}
    for h in history:
        if h['category'] in categories:
            categories[h['category']] += 1

    return render_template('index.html',
                           history=history,
                           total=total,
                           spam_count=spam_count,
                           ham_count=ham_count,
                           categories=categories)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Predict
    vector = vectorizer.transform([text])
    result = model.predict(vector)[0]
    prediction = 'Spam' if result == 1 else 'Not Spam'
    confidence = model.predict_proba(vector)[0][result] * 100

    # Simple category detection
    lower = text.lower()
    if "account" in lower or "verify" in lower or "login" in lower:
        category = "Phishing"
    elif "offer" in lower or "free" in lower or "win" in lower:
        category = "Promotion"
    else:
        category = "General"

    history.append({
        'text': text,
        'prediction': prediction,
        'confidence': f"{confidence:.2f}%",
        'category': category
    })

    return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    # Create a CSV in memory
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=['Text', 'Prediction', 'Confidence', 'Category'])
    writer.writeheader()

    # Write each history entry to the CSV
    for row in history:
        writer.writerow({
            'Text': row['text'],
            'Prediction': row['prediction'],
            'Confidence': row['confidence'],
            'Category': row['category']
        })

    # Move back to the start of the StringIO object
    output.seek(0)

    # Create a response to download the CSV
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
    response.headers["Content-Type"] = "text/csv"

    return response

if __name__ == '__main__':
    app.run(debug=True)
