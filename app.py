from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'best_model_rf.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Feature order must match training data column order (after dropping target)
FEATURES = [
    'Age',
    'Gender',
    'Education',
    'Employment_Status',
    'Income_Level',
    'Work_Stress_Level',
    'Financial_Stress',
    'Sleep_Hours_Night',
    'Physical_Activity_Hours_Week',
    'Screen_Time_Hours_Day',
    'Social_Media_Hours_Day',
    'Loneliness',
    'Social_Support',
    'Work_Life_Balance',
    'Feeling_Sad_Down',
    'Loss_Of_Interest',
    'Sleep_Trouble',
    'Fatigue',
    'Poor_Appetite_Or_Overeating',
    'Feeling_Worthless',
    'Concentration_Difficulty',
    'Anxious_Nervous',
    'Panic_Attacks',
    'Mood_Swings',
    'Irritability',
    'Obsessive_Thoughts',
    'Compulsive_Behavior',
    'Self_Harm_Thoughts',
    'Suicidal_Thoughts',
]

# Label encoding maps (matching LabelEncoder fit on training data — adjust if your encoder order differs)
GENDER_MAP      = {'Female': 0, 'Male': 1, 'Non-binary': 2, 'Other': 3}
EDUCATION_MAP   = {'Associate Degree': 0, "Bachelor's Degree": 1, 'High School': 2, 'Middle School': 3, "Master's Degree": 4, 'No Formal Education': 5, 'PhD': 6}
EMPLOYMENT_MAP  = {'Employed': 0, 'Freelancer': 1, 'Self-employed': 2, 'Student': 3, 'Unemployed': 4}
INCOME_MAP      = {'High': 0, 'Low': 1, 'Medium': 2}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Encode categoricals
        gender     = GENDER_MAP.get(data.get('Gender', ''), 1)
        education  = EDUCATION_MAP.get(data.get('Education', ''), 1)
        employment = EMPLOYMENT_MAP.get(data.get('Employment_Status', ''), 0)
        income     = INCOME_MAP.get(data.get('Income_Level', ''), 1)

        # Build feature vector in the exact column order
        features = [
            float(data.get('Age', 25)),
            gender,
            education,
            employment,
            income,
            int(data.get('Work_Stress_Level', 2)),          # 1–5 or 0–4
            int(data.get('Financial_Stress', 2)),
            float(data.get('Sleep_Hours_Night', 7)),
            float(data.get('Physical_Activity_Hours_Week', 3)),
            float(data.get('Screen_Time_Hours_Day', 4)),
            float(data.get('Social_Media_Hours_Day', 2)),
            int(data.get('Loneliness', 0)),                  # 0/1
            int(data.get('Social_Support', 1)),              # 0/1
            int(data.get('Work_Life_Balance', 1)),           # 0/1
            # Symptom flags (0/1)
            int(data.get('Feeling_Sad_Down', 0)),
            int(data.get('Loss_Of_Interest', 0)),
            int(data.get('Sleep_Trouble', 0)),
            int(data.get('Fatigue', 0)),
            int(data.get('Poor_Appetite_Or_Overeating', 0)),
            int(data.get('Feeling_Worthless', 0)),
            int(data.get('Concentration_Difficulty', 0)),
            int(data.get('Anxious_Nervous', 0)),
            int(data.get('Panic_Attacks', 0)),
            int(data.get('Mood_Swings', 0)),
            int(data.get('Irritability', 0)),
            int(data.get('Obsessive_Thoughts', 0)),
            int(data.get('Compulsive_Behavior', 0)),
            int(data.get('Self_Harm_Thoughts', 0)),
            int(data.get('Suicidal_Thoughts', 0)),
        ]

        X = np.array(features).reshape(1, -1)
        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0].tolist()

        # Map prediction label
        label = 'Mental Health Issue Detected' if prediction == 1 else 'No Mental Health Issue Detected'
        confidence = round(max(proba) * 100, 2)

        return jsonify({
            'prediction': prediction,
            'label': label,
            'confidence': confidence,
            'probability_no_issue': round(proba[0] * 100, 2),
            'probability_issue':    round(proba[1] * 100, 2),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
