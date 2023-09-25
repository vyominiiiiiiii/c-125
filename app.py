from flask import Flask,jsonify,request
from classifier import getPrediction

app=Flask(__name__)
@app.route("/predict-digit",methods=["POST"])
def predictData():
    img=request.files.get("digit")
    prediction=getPrediction(img)

    return jsonify({
        "prediction":prediction
    }),200

if __name__=="__main__":
    app.run(debug=True)