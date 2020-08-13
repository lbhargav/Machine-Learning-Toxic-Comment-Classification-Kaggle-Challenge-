import pickle

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_prob', methods=['POST'])
def predict_prob():
    # Loading our ML Model
    loaded_model = pickle.load(open('src/model.pkl', 'rb'))
    loaded_tfidf_vector = pickle.load(open('src/tfidf_vector.pkl', 'rb'))

    # Receives the input query from form
    if request.method == 'POST':
        user_comment = request.form['usercomment']
        user_comment_tfidf_vector = loaded_tfidf_vector.transform([user_comment])
        model_prediction = loaded_model.predict_proba(user_comment_tfidf_vector)[:, 1][0]
        model_prediction = str(round(model_prediction, 2))
    return render_template('results.html', prediction=model_prediction, name=user_comment)

if __name__ == '__main__':
    app.run(debug=True)