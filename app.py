from flask import Flask, render_template, request
import pickle
import numpy as np

import os
import pickle

# Get the folder where app.py is located
BASE_DIR = os.path.dirname(__file__)

# Load scalers and models using absolute paths
try:
    ms_path = os.path.join(BASE_DIR, "bestmodel_117.pk1")
    model_path = os.path.join(BASE_DIR, "xgb_model_117.pk1")

    ms = pickle.load(open(ms_path, "rb"))       # MinMaxScaler or StandardScaler
    model = pickle.load(open(model_path, "rb")) # XGBoost / RandomForest Model

except Exception as e:
    print("Error loading model or scaler:", e)

app = Flask(__name__)

@app.route('/')
def loadpage():
    return render_template("ecom.html")

@app.route('/y_predict', methods=["POST"])
def prediction():
    try:
        # Get form data
        Cost_of_the_Product = request.form["Cost_of_the_Product"]
        Discount_offered = request.form["Discount_offered"]
        Prior_purchases = request.form["Prior_purchases"]
        Weight_in_gms = request.form["Weight_in_gms"]
        Customer_rating = request.form["Customer_rating"]
        Customer_care_calls = request.form["Customer_care_calls"]


       

        # Prepare input data
        preds = [[
            float(Cost_of_the_Product), 
            float(Customer_rating), 
            int(Customer_care_calls), 
            int(Prior_purchases), 
            float(Discount_offered), 
            float(Weight_in_gms)
        ]]

        # Transform and predict
        transformed_preds = ms.transform(preds)
        prediction = model.predict(transformed_preds)
        prediction_proba = model.predict_proba(transformed_preds)[0]

        not_reach_prob = prediction_proba[0]
        reach_prob = prediction_proba[1]

        prediction_text = 'There is a {:.2f}% chance that your product will reach in time'.format(reach_prob * 100)
        print(prediction_text)
        print(prediction)

        return render_template("ecom.html", prediction_text=prediction_text)
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)


