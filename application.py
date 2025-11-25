# flask, pandas, scikit-learn, pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open(
    'LinearRegressionModel.pkl', 'rb'
))

# Load dataset
car = pd.read_csv("cleaned_car.csv")

@app.route("/")
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)  # newest first
    fuel_types = sorted(car['fuel_type'].unique())

    # {company: [models]}
    company_models = {}
    for company in companies:
        company_models[company] = sorted(
            car[car['company'] == company]['name'].unique()
        )

    return render_template(
        "index.html",
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types,
        company_models=company_models
    )

@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))

    # Keep same order as training
    prediction = model.predict(pd.DataFrame(
        [[car_model, company, year, fuel_type, kms_driven]],
        columns=['name', 'company', 'year', 'fuel_type', 'kms_driven']
    ))

    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
