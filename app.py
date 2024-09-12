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
    profile_pic = int(request.form['profile_pic'])
    username = request.form['username']
    fullname = request.form['fullname']
    name_equals_username = (lambda u, f: 1 if u == f else 0)(username, fullname)
    description = request.form['description']
    external_url = int(request.form['external_url'])
    private = int(request.form['private'])
    posts = int(request.form['posts'])
    followers = int(request.form['followers'])
    follows = int(request.form['follows'])

    username_length = len(username)
    fullname_length = len(fullname)
    fullname_words = len(fullname.split())
    description_length=len(description)

    features = [profile_pic, username_length, fullname_words, fullname_length,
                name_equals_username, description_length, external_url,
                private, posts, followers, follows]

    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = 'Fake Profile' if prediction[0] == 1 else 'Real Profile'

    return render_template('index.html', prediction_text=f'This profile is Predicted {output}')

if __name__ == "__main__":
    app.run(debug=True)
