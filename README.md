# Fake-OR-Real-Profile-Detection
This project is designed to identify fake social media profiles using machine learning. It leverages a logistic regression model to predict whether a profile is fake or real based on various features.The goal of this project is to provide a tool that can help in detecting potentially fake profiles by analyzing various profile attributes.
Fake Social Media Profile Detection
This project aims to detect fake social media profiles using a machine learning model built with logistic regression. The application features a user-friendly web interface developed using HTML, CSS, and Bootstrap, and is powered by a Flask backend to handle predictions.

Project Overview
Dataset Selection: The dataset used for training the model contains features that help in identifying fake social media profiles.
Model Creation: A logistic regression model was trained on the dataset and saved as a pickle file.
Frontend Development: The frontend is designed using HTML, CSS, and Bootstrap to provide a responsive user interface.
Flask Backend: Flask is used to handle form submissions and provide predictions based on the trained model.
Features
Profile Picture: Whether the profile has a profile picture.
Username Length: Length of the username.
Full Name: The full name of the profile.
Full Name Length: Length of the full name.
Name Equals Username: Whether the name is the same as the username.
Description Length: Length of the description.
External URL: Whether the profile contains an external URL.
Private Account: Whether the account is private.
Number of Posts: Total number of posts.
Number of Followers: Total number of followers.
Number of Follows: Total number of follows.
Getting Started
Dataset Selection
Dataset Features: The dataset includes the following features: profile picture, username length, full name words, full name length, name equals username, description length, external URL, private status, number of posts, number of followers, and number of follows.
Dataset Preparation: Clean and preprocess the data to ensure it is suitable for training the model.
Model Creation
Training the Model:

python
Copy code
import pickle
from sklearn.linear_model import LogisticRegression

# Example code for training and saving the model
model = LogisticRegression()
model.fit(X_train, y_train)  # X_train and y_train are your features and target
with open('forsm.pkl', 'wb') as file:
    pickle.dump(model, file)
Frontend Development
HTML: Create a form in HTML to collect user inputs for prediction.

CSS: Style the form using CSS for a better user experience.

Bootstrap: Use Bootstrap to make the form responsive and visually appealing.

html
Copy code
<!-- Example HTML Form -->
<form action="/predict" method="post">
    <label for="profile_pic">Profile Picture:</label>
    <select name="profile_pic">
        <option value="0">No</option>
        <option value="1">Yes</option>
    </select>
    <!-- Add more form fields as needed -->
    <button type="submit">Predict</button>
</form>
Flask Backend
Creating the Flask App:

python
Copy code
from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('forsm.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = 'Fake Profile' if prediction[0] == 1 else 'Real Profile'
    return render_template('index.html', prediction_text=f'This profile is likely a {output}')

if __name__ == "__main__":
    app.run(debug=True)
Running the Project
Install Dependencies: Make sure you have Flask, scikit-learn, and other necessary libraries installed.

bash
Copy code
pip install flask scikit-learn numpy
Run the Flask App: Start the Flask server to use the web application.

bash
Copy code
python app.py
Access the Application: Open your web browser and go to http://127.0.0.1:5000/ to use the application.
