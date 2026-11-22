from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("student_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    hours = float(data["study_hours"])

    result = model.predict([[hours]])

    return jsonify({
        "predicted_score": float(result[0])
    })
@app.route("/")
def home():
    return """
    <h1>welcome to my ai predictor!</h1>
    <p>your application is running successfully.</p>
    """
if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT",
                                10000))
    )
    
