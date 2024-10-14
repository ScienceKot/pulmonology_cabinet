from flask import Flask, request, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import transforms
from PIL import Image

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
    xray_image = db.Column(db.String(200), nullable=True)  # Store X-ray filename
    xray_tuberculosis_proba = db.Column(db.Float, nullable=True)  # Store predictions
    xray_covid_proba = db.Column(db.Float, nullable=True)  # Store predictions
    xray_normal_proba = db.Column(db.Float, nullable=True)  # Store predictions
    xray_bacterial_pneumonia_proba = db.Column(db.Float, nullable=True)  # Store predictions
    xray_viral_pneumonia_proba = db.Column(db.Float, nullable=True)  # Store predictions


# Create DB
with app.app_context():
    db.create_all()

# Load the pre-trained PyTorch model
# Replace 'YourModelClass' with the actual class name of your model
# and load the model from the appropriate checkpoint
model = torch.load('best-pulmonology-vgg19_bn-20_epoch-25088_relu-4-aug-256-Adam-0.001-bs64-x1.pth', map_location=torch.device('cpu'))
model.eval()

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

        return redirect(url_for('patient_list'))

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


# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
