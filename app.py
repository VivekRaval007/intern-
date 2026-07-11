from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        soi = float(request.form["SOi"])
        noi = float(request.form["Noi"])
        rspmi = float(request.form["RSPMi"])
        spmi = float(request.form["SPMi"])
        pmi = float(request.form["PMi"])

        features = np.array([[soi, noi, rspmi, spmi, pmi]])

        prediction = model.predict(features)

        result = round(prediction[0], 2)

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error : {e}")


if __name__ == "__main__":
    app.run(debug=True)