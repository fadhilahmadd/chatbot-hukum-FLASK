from flask import Flask, render_template, request, jsonify
from chatbot import chatbot_response, intents, model, words, lemmatizer

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    userText = request.json.get('msg')
    response = chatbot_response(userText, intents, model, words, lemmatizer)
    return jsonify({'answer': response})

if __name__ == "__main__":
    app.run(debug=True)
