from flask import Flask, render_template, request
from model import DreamJobModel

app = Flask(__name__)
model = DreamJobModel()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_input = request.form.get("skills")
        prediction = model.predict_job(user_input)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=False)

