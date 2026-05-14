from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
import json
import os
import hashlib
from functools import wraps
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests
from datetime import datetime, timedelta
import random
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
import matplotlib.pyplot as plt
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure secret key

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# OpenWeatherMap API configuration
WEATHER_API_KEY = 'faccd2a7e87da56ec8f5e37ab1ec6206'  # Replace with your OpenWeatherMap API key
WEATHER_API_URL = 'http://api.openweathermap.org/data/2.5/weather'

# Load the model
model = tf.keras.models.load_model('plant_disease_model.h5')

# Define disease classes (update these according to your model's classes)
CLASSES = [
    "Bacterial spot",
    "Early blight",
    "Late blight",
    "Leaf Mold",
    "Septoria leaf spot",
    "Spider mites",
    "Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato mosaic virus",
    "healthy"
]

# Add these constants near the top with other constants
ACTIVITY_FILE = os.path.join('data', 'recent_activity.json')

def preprocess_image(image):
    # Resize image to match model input size (update size according to your model)
    image = image.resize((128, 128))
    # Convert to array and preprocess
    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_disease_info(disease):
    # Comprehensive disease information
    cause = {
        "Bacterial spot": "Xanthomonas bacteria.",
        "Early blight": "Alternaria solani (fungus).",
        "Late blight": "Phytophthora infestans (fungus-like pathogen).",
        "Leaf Mold": "Passalora fulva (fungus).",
        "Septoria leaf spot": "Septoria lycopersici (fungus).",
        "Spider mites": "Tiny arachnids.",
        "Target Spot": "Corynespora cassiicola (fungus).",
        "Tomato Yellow Leaf Curl Virus": "Transmitted by whiteflies (Bemisia tabaci).",
        "Tomato mosaic virus": "Virus spread through contact, tools, and infected seeds",
        "healthy": " ",
    }

    symptoms = {
        "Bacterial spot": "Small, water-soaked spots on leaves and fruit that turn dark brown/black.",
        "Early blight": "Brown concentric rings (target-like spots) on lower leaves.",
        "Late blight": "Large, water-soaked lesions on leaves and stems; white moldy growth in humid conditions.",
        "Leaf Mold": "Yellow spots on upper leaf surfaces, velvety olive-green mold underneath.",
        "Septoria leaf spot": "Small, circular spots with gray centers and dark edges.",
        "Spider mites": "Yellowing, speckled leaves with fine webbing.",
        "Target Spot": "Brown lesions with concentric rings on leaves and fruit.",
        "Tomato Yellow Leaf Curl Virus": "Curling, yellowing of leaves, stunted growth.",
        "Tomato mosaic virus": "Mosaic-like mottling on leaves, leaf curling, and fruit distortion.",
        "healthy": "Deep green leaves, strong stems, and vibrant flowers.",
    }

    effects = {
        "Bacterial spot": "Can cause defoliation and reduce fruit yield.",
        "Early blight": "Causes defoliation, weakening the plant and reducing fruit production.",
        "Late blight": "Can spread rapidly and devastate crops.",
        "Leaf Mold": "Leads to leaf drop and reduced plant health.",
        "Septoria leaf spot": "Can cause severe defoliation, weakening the plant.",
        "Spider mites": "Saps plant nutrients, leading to leaf drop and reduced yields",
        "Target Spot": "Can reduce fruit quality and cause defoliation.",
        "Tomato Yellow Leaf Curl Virus": "Severely reduces fruit yield.",
        "Tomato mosaic virus": "Weakens the plant, reducing fruit production",
        "healthy": " ",
    }

    remedies = {
        "Bacterial spot": "Apply copper-based fungicides and avoid overhead watering.",
        "Early blight": "Remove infected leaves and apply neem oil or copper fungicide.",
        "Late blight": "Use fungicides like chlorothalonil and rotate crops.",
        "Leaf Mold": "Improve air circulation and use sulfur-based fungicides.",
        "Septoria leaf spot": "Prune affected leaves and apply organic fungicides.",
        "Spider mites": "Spray plants with neem oil or insecticidal soap.",
        "Target Spot": "Apply biofungicides and ensure proper irrigation.",
        "Tomato Yellow Leaf Curl Virus": "Use resistant varieties and control whiteflies.",
        "Tomato mosaic virus": "Remove infected plants and disinfect tools regularly.",
        "healthy": "No action needed. Keep monitoring for changes.",
    }

    return {
        'cause': cause.get(disease, 'No cause information available.'),
        'symptoms': symptoms.get(disease, 'No symptoms information available.'),
        'effects': effects.get(disease, 'No effects information available.'),
        'remedy': remedies.get(disease, 'No specific remedy available.')
    }

def predict_disease(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Make prediction
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Get disease name and information
    disease = CLASSES[predicted_class]
    disease_info = get_disease_info(disease)
    
    return {
        'disease': disease,
        'confidence': confidence,
        'cause': disease_info['cause'],
        'symptoms': disease_info['symptoms'],
        'effects': disease_info['effects'],
        'remedy': disease_info['remedy'],
        'status': 'success' if confidence > 0.5 else 'warning'
    }

# User data file path
USERS_FILE = 'data/users.json'
DISEASE_HISTORY_FILE = 'data/disease_history.json'

# Initialize files if they don't exist
if not os.path.exists('data'):
    os.makedirs('data')

if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)

if not os.path.exists(DISEASE_HISTORY_FILE):
    with open(DISEASE_HISTORY_FILE, 'w') as f:
        json.dump({}, f)

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def load_disease_history():
    """Load disease detection history from JSON file"""
    try:
        with open(DISEASE_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_disease_history(history):
    """Save disease detection history to JSON file"""
    with open(DISEASE_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def add_disease_detection(user_email, detection_result):
    """Add a new disease detection result to the history"""
    history = load_disease_history()
    
    # Initialize user history if not exists
    if user_email not in history:
        history[user_email] = []
    
    # Add new detection with timestamp
    detection_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'disease': detection_result['disease'],
        'confidence': detection_result['confidence'],
        'cause': detection_result['cause'],
        'symptoms': detection_result['symptoms'],
        'effects': detection_result['effects'],
        'remedy': detection_result['remedy']
    }
    
    # Add to beginning of list (most recent first)
    history[user_email].insert(0, detection_data)
    
    # Keep only last 50 detections per user
    history[user_email] = history[user_email][:50]
    
    save_disease_history(history)
    return detection_data

def get_user_disease_history(user_email, limit=7):
    """Get recent disease detection history for a user"""
    history = load_disease_history()
    if user_email not in history:
        return []
    return history[user_email][:limit]

def get_user_disease_distribution(user_email):
    """Get disease distribution for a user"""
    history = load_disease_history()
    if user_email not in history:
        return {'labels': [], 'data': []}
    
    # Count occurrences of each disease
    disease_counts = {}
    for detection in history[user_email]:
        disease = detection['disease']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Convert to lists for chart
    labels = list(disease_counts.keys())
    data = list(disease_counts.values())
    
    return {
        'labels': labels,
        'data': data
    }

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        users = load_users()
        if email in users and users[email]['password'] == hash_password(password):
            session['user'] = email
            session.permanent = True  # Make the session persistent
            log_activity(email, "User logged in", "Successful login")
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user' in session:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        city = request.form.get('city')
        state = request.form.get('state')
        country = request.form.get('country')
        
        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')
        
        users = load_users()
        if email in users:
            return render_template('register.html', error='Email already registered')
        
        users[email] = {
            'username': username,
            'password': hash_password(password),
            'city': city,
            'state': state,
            'country': country
        }
        save_users(users)
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    if 'user' in session:
        log_activity(session['user'], "User logged out", "User session ended")
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    users = load_users()
    user_data = users[session['user']]
    
    # Get recent activities for the user
    activity_history = load_activity_history()
    recent_activities = activity_history.get(session['user'], [])[:5]  # Get last 5 activities
    
    return render_template('dashboard.html', 
                         user=user_data,
                         recent_activities=recent_activities)

@app.route('/edit_profile', methods=['POST'])
@login_required
def edit_profile():
    users = load_users()
    user_email = session['user']
    
    # Get form data
    username = request.form.get('username')
    city = request.form.get('city')
    state = request.form.get('state')
    country = request.form.get('country')
    new_password = request.form.get('new_password')
    confirm_password = request.form.get('confirm_password')
    
    # Update user data
    users[user_email]['username'] = username
    users[user_email]['city'] = city
    users[user_email]['state'] = state
    users[user_email]['country'] = country
    
    # Update password if provided
    if new_password:
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return redirect(url_for('dashboard'))
        users[user_email]['password'] = hash_password(new_password)
    
    save_users(users)
    log_activity(user_email, "Profile Updated", "Updated user profile information")
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/disease_detection')
@login_required
def disease_detection():
    return render_template('disease_detection.html')

@app.route('/detect_disease', methods=['POST'])
@login_required
def detect_disease():
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image uploaded'
        })
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No image selected'
        })
    
    if file:
        try:
            # Save the image temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and process the image
            image = Image.open(filepath)
            processed_image = preprocess_image(image)
            
            # Get prediction
            prediction = model.predict(processed_image)
            disease_index = np.argmax(prediction[0])
            confidence = prediction[0][disease_index]
            
            # Get disease information
            disease = CLASSES[disease_index]
            disease_info = get_disease_info(disease)
            
            result = {
                'status': 'success',
                'disease': disease,
                'confidence': float(confidence),
                'cause': disease_info['cause'],
                'symptoms': disease_info['symptoms'],
                'effects': disease_info['effects'],
                'remedy': disease_info['remedy']
            }
            
            # Add detection to disease history
            add_disease_detection(session['user'], result)
            
            # Log the activity
            log_activity(session['user'], "Disease Detection", 
                        f"Detected {disease} with {confidence*100:.2f}% confidence")
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify(result)
        except Exception as e:
            # Clean up the temporary file if it exists
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'status': 'error',
                'message': f'Error processing image: {str(e)}'
            })

def get_weather_data(city, state, country):
    """Get weather data from OpenWeatherMap API"""
    try:
        # Construct location string
        location = f"{city},{state},{country}"
        
        # Make API request
        params = {
            'q': location,
            'appid': WEATHER_API_KEY,
            'units': 'metric'  # Use metric units
        }
        
        response = requests.get(WEATHER_API_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            # Extract weather data with proper error handling
            weather_data = {
                'temperature': round(data.get('main', {}).get('temp', 0)),
                'humidity': data.get('main', {}).get('humidity', 0),
                'wind_speed': round(data.get('wind', {}).get('speed', 0) * 3.6),  # Convert m/s to km/h
                'description': data.get('weather', [{}])[0].get('description', 'Unknown'),
                'rain_probability': data.get('rain', {}).get('1h', 0) * 100  # Convert to percentage
            }
            
            # Ensure all values are within reasonable ranges
            weather_data['temperature'] = max(-50, min(50, weather_data['temperature']))
            weather_data['humidity'] = max(0, min(100, weather_data['humidity']))
            weather_data['wind_speed'] = max(0, min(200, weather_data['wind_speed']))
            weather_data['rain_probability'] = max(0, min(100, weather_data['rain_probability']))
            
            return weather_data
        else:
            print(f"Weather API error: {data.get('message', 'Unknown error')}")
            return get_default_weather_data()
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return get_default_weather_data()

def get_default_weather_data():
    """Return default weather data when API fails"""
    return {
        'temperature': 25,
        'humidity': 60,
        'wind_speed': 10,
        'description': 'Partly cloudy',
        'rain_probability': 30
    }

def calculate_crop_conditions(weather_data):
    """Calculate crop growth conditions based on weather data"""
    if not weather_data:
        weather_data = get_default_weather_data()
    
    # Temperature suitability (optimal range is 20-30°C)
    temp = weather_data['temperature']
    temp_score = max(0, min(100, (temp - 10) * 10))  # Simple linear scoring
    
    # Moisture level (based on humidity and rain probability)
    humidity = weather_data['humidity']
    rain_prob = weather_data['rain_probability']
    moisture_score = (humidity + rain_prob) / 2
    
    # Overall growth potential (weighted average)
    growth_potential = (temp_score * 0.4 + moisture_score * 0.6)
    
    return {
        'temperature_score': round(temp_score),
        'moisture_score': round(moisture_score),
        'growth_potential': round(growth_potential)
    }

def get_disease_history():
    """Get historical disease detection data"""
    # This would typically come from a database
    # For now, we'll generate sample data
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    confidence = [round(random.uniform(0.6, 0.95), 2) for _ in range(7)]
    
    return {
        'dates': dates,
        'confidence': confidence
    }

def get_disease_distribution():
    """Get distribution of detected diseases"""
    # This would typically come from a database
    # For now, we'll generate sample data
    diseases = CLASSES[:5]  # Use first 5 diseases as example
    counts = [random.randint(1, 10) for _ in range(5)]
    
    return {
        'labels': diseases,
        'data': counts
    }

def get_weather_forecast():
    """Get 7-day weather forecast"""
    try:
        # Get current weather data first
        current_weather = get_weather_data('London', 'England', 'UK')  # Replace with user's location
        
        # Generate forecast data starting with current weather
        dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
        temperatures = [current_weather['temperature']]  # Start with current temperature
        rain_probability = [current_weather['rain_probability']]  # Start with current rain probability
        
        # Generate variations for remaining days
        for _ in range(6):
            # Temperature variation: ±5°C from current
            temp_variation = random.uniform(-5, 5)
            temperatures.append(round(current_weather['temperature'] + temp_variation))
            
            # Rain probability variation: ±20% from current
            rain_variation = random.uniform(-20, 20)
            rain_prob = current_weather['rain_probability'] + rain_variation
            rain_probability.append(max(0, min(100, round(rain_prob))))
        
        return {
            'dates': dates,
            'temperatures': temperatures,
            'rain_probability': rain_probability
        }
    except Exception as e:
        print(f"Error generating weather forecast: {e}")
        return get_default_weather_forecast()

def get_default_weather_forecast():
    """Return default weather forecast data when API fails"""
    current_date = datetime.now()
    dates = [(current_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
    temperatures = [25] * 7  # Default temperature
    rain_probability = [30] * 7  # Default rain probability
    
    return {
        'dates': dates,
        'temperatures': temperatures,
        'rain_probability': rain_probability
    }

@app.route('/analytics')
@login_required
def analytics():
    users = load_users()
    user_data = users[session['user']]
    
    # Get weather data for user's location with error handling
    weather_data = get_weather_data(user_data['city'], user_data['state'], user_data['country'])
    
    # Calculate crop conditions
    crop_conditions = calculate_crop_conditions(weather_data)
    
    # Get user-specific disease history and distribution
    disease_history = get_user_disease_history(session['user'])
    disease_distribution = get_user_disease_distribution(session['user'])
    
    # Format disease history for chart
    formatted_history = {
        'dates': [],
        'confidence': []
    }
    
    if disease_history:
        formatted_history['dates'] = [detection['timestamp'].split()[0] for detection in disease_history]
        formatted_history['confidence'] = [float(detection['confidence']) for detection in disease_history]
    
    # Format disease distribution for chart
    formatted_distribution = {
        'labels': [],
        'data': []
    }
    
    if disease_distribution['labels'] and disease_distribution['data']:
        formatted_distribution['labels'] = disease_distribution['labels']
        formatted_distribution['data'] = disease_distribution['data']
    
    # Get weather forecast using user's location
    weather_forecast = get_weather_forecast()
    
    # Ensure current weather matches first day of forecast
    weather_forecast['temperatures'][0] = weather_data['temperature']
    weather_forecast['rain_probability'][0] = weather_data['rain_probability']
    
    return render_template('analytics.html',
                         weather=weather_data,
                         crop_conditions=crop_conditions,
                         disease_history=formatted_history,
                         disease_distribution=formatted_distribution,
                         weather_forecast=weather_forecast)

def create_disease_history_chart(dates, confidence):
    """Create a line chart for disease detection history"""
    plt.figure(figsize=(8, 4))
    plt.plot(dates, confidence, marker='o', linewidth=2, color='#4BC0C0')
    plt.title('Disease Detection Confidence Over Time')
    plt.xlabel('Date')
    plt.ylabel('Confidence')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def create_disease_distribution_chart(labels, data):
    """Create a pie chart for disease distribution"""
    plt.figure(figsize=(8, 8))
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
    plt.pie(data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Detected Diseases')
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def create_weather_forecast_chart(dates, temperatures, rain_probability):
    """Create a line chart for weather forecast"""
    plt.figure(figsize=(10, 4))
    plt.plot(dates, temperatures, marker='o', linewidth=2, color='#FF6384', label='Temperature (°C)')
    plt.plot(dates, rain_probability, marker='o', linewidth=2, color='#36A2EB', label='Rain Probability (%)')
    plt.title('7-Day Weather Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(rotation=45)
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

@app.route('/generate_report')
@login_required
def generate_report():
    """Generate and download a PDF report of analytics data"""
    users = load_users()
    user_data = users[session['user']]
    
    # Get all the necessary data
    weather_data = get_weather_data(user_data['city'], user_data['state'], user_data['country'])
    crop_conditions = calculate_crop_conditions(weather_data)
    disease_history = get_user_disease_history(session['user'])
    disease_distribution = get_user_disease_distribution(session['user'])
    weather_forecast = get_weather_forecast()
    
    # Create a temporary file for the PDF
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f'analytics_report_{session["user"]}.pdf')
    
    # Create the PDF document
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    elements.append(Paragraph(f"SmartFarmAI Analytics Report", title_style))
    
    # Subtitle with user info
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=20,
        alignment=1
    )
    elements.append(Paragraph(f"Generated for {user_data['username']}", subtitle_style))
    
    # Date
    date_style = ParagraphStyle(
        'Date',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.gray,
        alignment=1
    )
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style))
    elements.append(Spacer(1, 30))
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", styles['Heading2']))
    summary_text = f"""
    This report provides a comprehensive analysis of your farm's current status, including weather conditions, 
    crop health, and disease detection history. The data shows that your crops are experiencing {weather_data['description']} 
    conditions with a {crop_conditions['growth_potential']}% overall growth potential.
    """
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Weather Information
    elements.append(Paragraph("Current Weather Conditions", styles['Heading2']))
    weather_data_table = [
        ["Parameter", "Value"],
        ["Temperature", f"{weather_data['temperature']}°C"],
        ["Humidity", f"{weather_data['humidity']}%"],
        ["Wind Speed", f"{weather_data['wind_speed']} km/h"],
        ["Description", weather_data['description']],
        ["Rain Probability", f"{weather_data['rain_probability']}%"]
    ]
    
    weather_table = Table(weather_data_table, colWidths=[2*inch, 2*inch])
    weather_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(weather_table)
    elements.append(Spacer(1, 20))
    
    # Weather Forecast Chart
    elements.append(Paragraph("7-Day Weather Forecast", styles['Heading2']))
    forecast_chart = create_weather_forecast_chart(
        weather_forecast['dates'],
        weather_forecast['temperatures'],
        weather_forecast['rain_probability']
    )
    elements.append(RLImage(forecast_chart, width=6*inch, height=3*inch))
    elements.append(Spacer(1, 20))
    
    # Crop Conditions
    elements.append(Paragraph("Crop Growth Conditions", styles['Heading2']))
    crop_data = [
        ["Condition", "Score", "Status"],
        ["Temperature Suitability", f"{crop_conditions['temperature_score']}%", 
         "Optimal" if crop_conditions['temperature_score'] > 70 else "Suboptimal"],
        ["Moisture Level", f"{crop_conditions['moisture_score']}%",
         "Adequate" if crop_conditions['moisture_score'] > 60 else "Insufficient"],
        ["Overall Growth Potential", f"{crop_conditions['growth_potential']}%",
         "Good" if crop_conditions['growth_potential'] > 70 else "Fair"]
    ]
    
    crop_table = Table(crop_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    crop_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(crop_table)
    elements.append(Spacer(1, 20))
    
    # Disease Detection Analysis
    elements.append(Paragraph("Disease Detection Analysis", styles['Heading2']))
    
    # Disease History Chart
    if disease_history:
        elements.append(Paragraph("Disease Detection History", styles['Heading3']))
        history_chart = create_disease_history_chart(
            [detection['timestamp'].split()[0] for detection in disease_history],
            [float(detection['confidence']) for detection in disease_history]
        )
        elements.append(RLImage(history_chart, width=6*inch, height=3*inch))
        elements.append(Spacer(1, 20))
    
    # Disease Distribution Chart
    if disease_distribution['labels'] and disease_distribution['data']:
        elements.append(Paragraph("Disease Distribution", styles['Heading3']))
        distribution_chart = create_disease_distribution_chart(
            disease_distribution['labels'],
            disease_distribution['data']
        )
        elements.append(RLImage(distribution_chart, width=6*inch, height=4*inch))
        elements.append(Spacer(1, 20))
        
        # Detailed Disease Information
        elements.append(Paragraph("Recent Disease Detections", styles['Heading3']))
        disease_data = [["Date", "Disease", "Confidence", "Status"]]
        for detection in disease_history:
            status = "High Risk" if detection['confidence'] > 0.8 else "Medium Risk" if detection['confidence'] > 0.5 else "Low Risk"
            disease_data.append([
                detection['timestamp'].split()[0],
                detection['disease'],
                f"{detection['confidence']*100:.1f}%",
                status
            ])
        
        disease_table = Table(disease_data, colWidths=[1.5*inch, 2*inch, 1.5*inch, 1.5*inch])
        disease_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(disease_table)
    else:
        elements.append(Paragraph("No disease detections recorded.", styles['Normal']))
    
    # Recommendations
    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Recommendations", styles['Heading2']))
    recommendations = []
    
    # Weather-based recommendations
    if weather_data['rain_probability'] > 70:
        recommendations.append("High rain probability detected. Consider implementing additional drainage measures.")
    elif weather_data['rain_probability'] < 30:
        recommendations.append("Low rain probability. Ensure proper irrigation systems are in place.")
    
    # Temperature-based recommendations
    if weather_data['temperature'] > 30:
        recommendations.append("High temperatures detected. Consider implementing shade structures.")
    elif weather_data['temperature'] < 15:
        recommendations.append("Low temperatures detected. Consider using protective covers.")
    
    # Disease-based recommendations
    if disease_history:
        recent_detections = [d for d in disease_history if float(d['confidence']) > 0.7]
        if recent_detections:
            recommendations.append("High-confidence disease detections found. Review and implement recommended treatments.")
    
    if recommendations:
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", styles['Normal']))
    else:
        elements.append(Paragraph("No specific recommendations at this time. Continue regular monitoring.", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    
    # Log the activity before sending the file
    log_activity(session['user'], "Analytics Report Generated", 
                "Generated comprehensive analytics report in PDF format")
    
    # Send the file to the user
    return send_file(
        pdf_path,
        as_attachment=True,
        download_name=f'analytics_report_{datetime.now().strftime("%Y%m%d")}.pdf',
        mimetype='application/pdf'
    )

@app.route('/recommendations')
@login_required
def recommendations():
    """Display plant care recommendations based on weather and disease data"""
    users = load_users()
    user_data = users[session['user']]
    
    # Get weather data for user's location
    weather_data = get_weather_data(user_data['city'], user_data['state'], user_data['country'])
    
    # Get disease history
    disease_history = get_user_disease_history(session['user'])
    
    return render_template('recommendations.html',
                         weather=weather_data,
                         disease_history=disease_history)

def load_activity_history():
    """Load activity history from JSON file"""
    if os.path.exists(ACTIVITY_FILE):
        try:
            with open(ACTIVITY_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_activity_history(history):
    """Save activity history to JSON file"""
    with open(ACTIVITY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

def log_activity(user_email, activity, details):
    """Log a new activity for a user"""
    history = load_activity_history()
    
    # Create new activity entry
    new_activity = {
        "id": f"ACT{len(history.get(user_email, [])) + 1:03d}",
        "activity": activity,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details
    }
    
    # Initialize user's activity list if it doesn't exist
    if user_email not in history:
        history[user_email] = []
    
    # Add new activity and keep only the last 50 activities
    history[user_email].insert(0, new_activity)
    history[user_email] = history[user_email][:50]
    
    # Save updated history
    save_activity_history(history)

if __name__ == '__main__':
    # Run the app on all available network interfaces (0.0.0.0)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True) 