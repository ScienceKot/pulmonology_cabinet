from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import random
from datetime import datetime
import pickle

# Initialize Flask app
app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = r'static\uploads'
db = SQLAlchemy(app)

# Model for the patients
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    surname = db.Column(db.String(100), nullable=False)
    patient_id = db.Column(db.String(10), unique=True, nullable=False)
    birth_date = db.Column(db.String(10), nullable=False)

    # X-ray related fields
    xray_image = db.Column(db.String(200), nullable=True)  # Store X-ray filename
    xray_tuberculosis_proba = db.Column(db.Float, nullable=True)  # Tuberculosis prediction probability
    xray_covid_proba = db.Column(db.Float, nullable=True)  # COVID-19 prediction probability
    xray_normal_proba = db.Column(db.Float, nullable=True)  # Normal prediction probability
    xray_bacterial_pneumonia_proba = db.Column(db.Float, nullable=True)  # Bacterial pneumonia probability
    xray_viral_pneumonia_proba = db.Column(db.Float, nullable=True)  # Viral pneumonia probability

    # Lung cancer test fields
    gender = db.Column(db.String(1), nullable=True)  # M or F
    age = db.Column(db.Integer, nullable=True)
    smoking = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    yellow_fingers = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    anxiety = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    peer_pressure = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    chronic_disease = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    fatigue = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    allergy = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    wheezing = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    alcohol = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    coughing = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    shortness_of_breath = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    swallowing_difficulty = db.Column(db.Integer, nullable=True)  # YES=2, NO=1
    chest_pain = db.Column(db.Integer, nullable=True)  # YES=2, NO=1

    lung_cancer_probability = db.Column(db.Float, nullable=True)  # Computed probability


# Create DB
with app.app_context():
    db.create_all()

# Load the pre-trained PyTorch model
# Replace 'YourModelClass' with the actual class name of your model
# and load the model from the appropriate checkpoint
model = torch.load('best-pulmonology-vgg19_bn-20_epoch-25088_relu-4-aug-256-Adam-0.001-bs64-x1.pth', map_location=torch.device('cpu'))
model.eval()

lung_cancer_model = pickle.load(open("trained_pipeline.pkl", "rb"))

# Define class names
class_names = ['Tuberculosis', 'Corona Virus Disease', 'Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']

# Image preprocessing pipeline (resize, normalize, convert to tensor)
transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.ConvertImageDtype(torch.float32),
    ])


# 1. Home - GET only
@app.route('/')
def home():
    return render_template("home.html")


# 2. Add patient - GET and POST
@app.route('/add_patient', methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        name = request.form['name']
        surname = request.form['surname']
        patient_id = request.form['patient_id']
        birth_date = request.form['birth_date']

        # Preprocess the data
        patient = Patient(name=name, surname=surname, patient_id=patient_id, birth_date=birth_date)
        db.session.add(patient)
        db.session.commit()

        return redirect(url_for('lung_cancer_test', id=patient.id))

    return render_template("add_patient.html")


# 3. Patient list - GET only
@app.route('/patients', methods=['GET'])
def patient_list():
    patients = Patient.query.all()
    return render_template('patient_list.html', patients=patients)


# 4. Patient/id - GET and POST (edit patient)
@app.route('/patient/<int:id>', methods=['GET', 'POST'])
def patient_detail(id):
    patient = Patient.query.get_or_404(id)

    if request.method == 'POST':
        patient.name = request.form['name']
        patient.surname = request.form['surname']
        patient.birth_date = request.form['birth_date']
        db.session.commit()
        return redirect(url_for('patient_list'))

    return render_template('patient_detail.html', patient=patient)


# 5. Upload X-ray page - GET and POST
@app.route('/upload_xray/<int:id>', methods=['GET', 'POST'])
def upload_xray(id):
    patient = Patient.query.get_or_404(id)

    if request.method == 'POST':
        # Get the file from the form
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(file_path)
            file.save(file_path)

            # Save file reference in database
            patient.xray_image = filename

            # Load and preprocess the image
            image = Image.open(file_path).convert('RGB')
            image = transform(image)
            image = image.unsqueeze(0)  # Add batch dimension

            # Run the image through the model
            with torch.no_grad():
                outputs = model(image)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                prediction = {class_names[i]: probabilities[i].item() for i in range(len(class_names))}

            # Store prediction probabilities in the database
            patient.xray_tuberculosis_proba = prediction["Tuberculosis"]
            patient.xray_covid_proba = prediction["Corona Virus Disease"]
            patient.xray_normal_proba = prediction["Normal"]
            patient.xray_bacterial_pneumonia_proba = prediction["Bacterial Pneumonia"]
            patient.xray_viral_pneumonia_proba = prediction["Viral Pneumonia"]
            db.session.commit()

            return render_template('xray_result.html', prediction=prediction, image=filename)

    return render_template("upload_xray.html")


def calculate_age(birth_date_str):
    # Convert the birth_date_str (which is a string) into a datetime object
    birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d')

    # Get the current date
    today = datetime.today()

    # Calculate the age
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

    return age

@app.route('/patient/<int:id>/lung_cancer_test', methods=['GET', 'POST'])
def lung_cancer_test(id):
    patient = Patient.query.get_or_404(id)

    if request.method == 'POST':
        # Capture form inputs and store them in the database
        patient.gender = request.form['gender']
        patient.age = calculate_age(patient.birth_date)
        patient.smoking = int(request.form['smoking'])
        patient.yellow_fingers = int(request.form['yellow_fingers'])
        patient.anxiety = int(request.form['anxiety'])
        patient.peer_pressure = int(request.form['peer_pressure'])
        patient.chronic_disease = int(request.form['chronic_disease'])
        patient.fatigue = int(request.form['fatigue'])
        patient.allergy = int(request.form['allergy'])
        patient.wheezing = int(request.form['wheezing'])
        patient.alcohol = int(request.form['alcohol'])
        patient.coughing = int(request.form['coughing'])
        patient.shortness_of_breath = int(request.form['shortness_of_breath'])
        patient.swallowing_difficulty = int(request.form['swallowing_difficulty'])
        patient.chest_pain = int(request.form['chest_pain'])

        X = [[
            patient.gender, patient.age, patient.smoking, patient.yellow_fingers, patient.anxiety, patient.peer_pressure,
            patient.chronic_disease, patient.fatigue, patient.allergy, patient.wheezing, patient.alcohol,
            patient.coughing, patient.shortness_of_breath, patient.swallowing_difficulty, patient.chest_pain
        ]]
        df = pd.DataFrame(X, columns=["GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY", "PEER_PRESSURE", "CHRONIC DISEASE",
                                      "FATIGUE ", "ALLERGY ", "WHEEZING", "ALCOHOL CONSUMING", "COUGHING", "SHORTNESS OF BREATH",
                                      "SWALLOWING DIFFICULTY", "CHEST PAIN"])

        patient.lung_cancer_probability = lung_cancer_model.predict_proba(df)[0][1]
        db.session.commit()

        return redirect(url_for('patient_detail', id=patient.id))

    return render_template('lung_cancer_test.html', patient=patient)


# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
