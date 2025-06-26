from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from generate_dataset import generate_attendance_data
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Define feature lists and their allowed values
categorical_features = {
    'health_status': ['Good', 'Fair', 'Poor'],
    'transportation_mode': ['Bus', 'Car', 'Walking', 'Bicycle'],
    'extracurricular_activities': ['Yes', 'No'],
    'class_time': ['Morning', 'Afternoon', 'Evening'],
    'subject_difficulty': ['Easy', 'Medium', 'Hard']
}

numerical_features = [
    'previous_attendance',
    'previous_grades',
    'study_hours',
    'distance',
    'sleep_hours',
    'previous_absence_count'
]

# Global variables for model and encoders
model = None
label_encoders = {}
target_encoder = None
scaler = StandardScaler()
feature_importances = None
feature_names = None

# Load and preprocess data
def load_data():
    if not os.path.exists('attendance.csv'):
        # If dataset doesn't exist, generate it
        df = generate_attendance_data(1000)
        df.to_csv('attendance.csv', index=False)
    return pd.read_csv('attendance.csv')

# Train model
def train_model():
    global model, label_encoders, target_encoder, scaler
    
    df = load_data()
    
    # Convert categorical variables to numerical
    for col, allowed_values in categorical_features.items():
        le = LabelEncoder()
        # Fit the encoder with all possible values
        le.fit(allowed_values)
        df[col] = le.transform(df[col])
        label_encoders[col] = le
    
    # Scale numerical features
    for col in numerical_features:
        scaler.fit_transform(df[[col]])
    
    # Prepare features and target
    X = df.drop(['attendance_status'], axis=1)
    y = df['attendance_status']
    
    # Convert target to numerical
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    target_encoder = le_target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and encoders
    joblib.dump(model, 'attendance_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(le_target, 'target_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, label_encoders, le_target, scaler

# Initialize model and encoders when the application starts
try:
    if os.path.exists('attendance_model.pkl'):
        model = joblib.load('attendance_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        scaler = joblib.load('scaler.pkl')
    else:
        model, label_encoders, target_encoder, scaler = train_model()
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    model, label_encoders, target_encoder, scaler = train_model()

def load_model_and_encoders():
    """Load the model and encoders with error handling."""
    global model, label_encoders, feature_importances, feature_names
    
    try:
        # Load model
        model_path = 'model/student_attendance_model.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully")
        
        # Load encoders
        encoder_files = {
            'health_status': 'model/health_status_encoder.pkl',
            'transportation_mode': 'model/transportation_mode_encoder.pkl',
            'extracurricular_activities': 'model/extracurricular_activities_encoder.pkl',
            'class_time': 'model/class_time_encoder.pkl',
            'subject_difficulty': 'model/subject_difficulty_encoder.pkl'
        }
        
        for feature, file_path in encoder_files.items():
            if os.path.exists(file_path):
                label_encoders[feature] = joblib.load(file_path)
                logging.info(f"Loaded encoder for {feature}")
            else:
                logging.warning(f"Encoder file not found for {feature}")
                # Create a default encoder if file not found
                label_encoders[feature] = LabelEncoder()
                label_encoders[feature].fit(['default'])
        
        # Load feature importances if available
        if os.path.exists('model/feature_importances.pkl'):
            feature_importances = joblib.load('model/feature_importances.pkl')
            feature_names = joblib.load('model/feature_names.pkl')
            logging.info("Feature importances loaded successfully")
        else:
            # Set default feature names based on the model
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
            logging.warning("Feature importances file not found, using default feature names")
        
    except Exception as e:
        logging.error(f"Error loading model or encoders: {str(e)}")
        raise

def validate_input(data):
    """Validate input data and return normalized values"""
    validated_data = {}
    
    # Validate categorical features
    for feature, allowed_values in categorical_features.items():
        if feature not in data:
            raise ValueError(f"Missing required field: {feature}")
        if data[feature] not in allowed_values:
            raise ValueError(f"Invalid value for {feature}. Allowed values: {allowed_values}")
        validated_data[feature] = data[feature]
    
    # Validate numerical features
    numerical_features = {
        'previous_attendance': (0, 1),
        'study_hours': (0, 1),
        'sleep_hours': (0, 1),
        'previous_grades': (0, 1),
        'distance': (0, 1),
        'previous_absence_count': (0, 1)
    }
    
    for feature, (min_val, max_val) in numerical_features.items():
        if feature not in data:
            raise ValueError(f"Missing required field: {feature}")
        try:
            value = float(data[feature])
            if not min_val <= value <= max_val:
                raise ValueError(f"{feature} must be between {min_val} and {max_val}")
            validated_data[feature] = value
        except ValueError as e:
            raise ValueError(f"Invalid value for {feature}: {str(e)}")
    
    return validated_data

def prepare_features(data):
    """Prepare features for prediction by encoding categorical variables and normalizing numerical ones"""
    features = []
    
    # Encode categorical features
    for feature in categorical_features:
        if feature in label_encoders:
            value = data[feature]
            if value not in label_encoders[feature].classes_:
                # Handle unseen labels by using the most common class
                value = label_encoders[feature].classes_[0]
            encoded_value = label_encoders[feature].transform([value])[0]
            features.append(encoded_value)
    
    # Add numerical features
    for feature in numerical_features:
        features.append(data[feature])
    
    return features

def analyze_risk_factors(data):
    """Analyze risk factors based on input data with detailed insights"""
    risk_factors = []
    risk_level = "Low"  # Default risk level
    
    # Health and Well-being Analysis
    health_risks = []
    if data['health_status'] == 'Poor':
        health_risks.append("Poor health status")
    if data['sleep_hours'] < 0.25:  # Less than 6 hours
        health_risks.append("Insufficient sleep")
    elif data['sleep_hours'] > 0.4:  # More than 9.6 hours
        health_risks.append("Excessive sleep")
    if health_risks:
        risk_factors.append({
            "category": "Health & Well-being",
            "risks": health_risks,
            "impact": "High" if len(health_risks) > 1 else "Medium"
        })
        risk_level = "High" if len(health_risks) > 1 else "Medium"

    # Academic Performance Analysis
    academic_risks = []
    if data['previous_grades'] < 0.6:
        academic_risks.append("Low previous grades")
    if data['study_hours'] < 0.2:  # Less than 5 hours
        academic_risks.append("Low study hours")
    if data['subject_difficulty'] == 'Hard':
        academic_risks.append("Challenging subject difficulty")
    if academic_risks:
        risk_factors.append({
            "category": "Academic Performance",
            "risks": academic_risks,
            "impact": "High" if len(academic_risks) > 1 else "Medium"
        })
        if risk_level != "High":
            risk_level = "High" if len(academic_risks) > 1 else "Medium"

    # Attendance History Analysis
    attendance_risks = []
    if data['previous_attendance'] < 0.7:
        attendance_risks.append("Low previous attendance record")
    if data['previous_absence_count'] > 0.2:
        attendance_risks.append("High number of previous absences")
    if attendance_risks:
        risk_factors.append({
            "category": "Attendance History",
            "risks": attendance_risks,
            "impact": "High" if len(attendance_risks) > 1 else "Medium"
        })
        if risk_level != "High":
            risk_level = "High" if len(attendance_risks) > 1 else "Medium"

    # Environmental Factors Analysis
    environmental_risks = []
    if data['transportation_mode'] == 'Walking' and data['distance'] > 0.3:
        environmental_risks.append("Long walking distance")
    if data['transportation_mode'] == 'Bicycle' and data['distance'] > 0.4:
        environmental_risks.append("Long cycling distance")
    if environmental_risks:
        risk_factors.append({
            "category": "Environmental Factors",
            "risks": environmental_risks,
            "impact": "High" if len(environmental_risks) > 1 else "Medium"
        })
        if risk_level != "High":
            risk_level = "High" if len(environmental_risks) > 1 else "Medium"

    # Transportation Analysis
    transport_risks = []
    if data['transportation_mode'] == 'Walking' and data['distance'] > 0.3:
        transport_risks.append("Long walking distance")
    if data['transportation_mode'] == 'Bicycle' and data['distance'] > 0.4:
        transport_risks.append("Long cycling distance")
    if transport_risks:
        risk_factors.append({
            "category": "Transportation",
            "risks": transport_risks,
            "impact": "High" if len(transport_risks) > 1 else "Medium"
        })
        if risk_level != "High":
            risk_level = "High" if len(transport_risks) > 1 else "Medium"

    # Add recommendations based on risk factors
    recommendations = []
    if health_risks:
        recommendations.append("Consider consulting a healthcare provider")
    if academic_risks:
        recommendations.append("Seek additional academic support or tutoring")
    if attendance_risks:
        recommendations.append("Develop a regular attendance routine")
    if environmental_risks:
        recommendations.append("Engage with school support services")
    if transport_risks:
        recommendations.append("Explore alternative transportation options")

    return {
        "risk_factors": risk_factors,
        "risk_level": risk_level,
        "recommendations": recommendations
    }

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/predict')
def predict_page():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input data
        if not validate_input(data):
            return jsonify({'error': 'Invalid input data'}), 400
        
        # Prepare features for prediction
        features = prepare_features(data)
        
        # Get prediction probabilities
        probabilities = model.predict_proba([features])[0]
        
        # Map probabilities to classes (0: Absent, 1: Present)
        present_prob = float(probabilities[1])  # Probability of class 1 (Present)
        absent_prob = float(probabilities[0])   # Probability of class 0 (Absent)
        
        # Determine prediction based on highest probability
        prediction_label = "Present" if present_prob > absent_prob else "Absent"
        
        # Get detailed risk analysis
        risk_analysis = analyze_risk_factors(data)
        
        # Calculate confidence score based on the predicted class probability
        confidence_score = present_prob * 100 if prediction_label == "Present" else absent_prob * 100
        
        # Prepare response
        response = {
            'prediction': prediction_label,
            'probabilities': {
                'Present': round(present_prob * 100, 1),
                'Absent': round(absent_prob * 100, 1)
            },
            'confidence_score': round(confidence_score, 1),
            'risk_analysis': risk_analysis
        }
        
        # Log prediction details
        app.logger.info(f"Prediction made: {prediction_label} with confidence {confidence_score}%")
        app.logger.info(f"Probabilities - Present: {present_prob*100}%, Absent: {absent_prob*100}%")
        app.logger.info(f"Risk level: {risk_analysis['risk_level']}")
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    try:
        df = load_data()
        features = df.drop(['attendance_status', 'student_id', 'date'], axis=1).columns.tolist()
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    try:
        df = load_data()
        model_info = {
            'total_samples': len(df),
            'feature_names': df.drop(['attendance_status', 'student_id', 'date'], axis=1).columns.tolist(),
            'target_distribution': df['attendance_status'].value_counts().to_dict(),
            'feature_types': {
                'numerical': numerical_features,
                'categorical': list(categorical_features.keys())
            }
        }
        return jsonify({
            'success': True,
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    try:
        load_model_and_encoders()
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Application startup error: {str(e)}")
        raise 