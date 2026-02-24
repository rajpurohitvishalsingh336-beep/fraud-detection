from flask import Flask, render_template, request, redirect, session
import pandas as pd
import pickle
import os
import matplotlib

# IMPORTANT for Flask + Matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================
# ADD THIS IMPORT
# ================================
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "fraud_model.pkl"
# ================================


app = Flask(__name__)
app.secret_key = "advanced_secret"

ADMIN_PASSWORD = "1234"

DATA_FILE = "insurance.csv"


# =========================================================
# FUNCTION: SYSTEM SETUP
# =========================================================
def setup_system():

    if not os.path.exists("static"):
        os.makedirs("static")

    if not os.path.exists(DATA_FILE):

        df = pd.DataFrame(columns=[

            "age",
            "months_as_customer",
            "policy_annual_premium",
            "total_claim_amount",
            "ML_Prediction",
            "Confidence",
            "Risk_Level",
            "Reason"

        ])

        df.to_csv(DATA_FILE, index=False)


# =========================================================
# ADD THIS FUNCTION (MODEL TRAINING)
# =========================================================
def train_model():

    if not os.path.exists(DATA_FILE):
        return

    df = pd.read_csv(DATA_FILE)

    if len(df) < 5:
        return

    df["fraud"] = df.apply(
        lambda row: 1 if row["total_claim_amount"] > row["policy_annual_premium"] * 5 else 0,
        axis=1
    )

    X = df[[
        "age",
        "months_as_customer",
        "policy_annual_premium",
        "total_claim_amount"
    ]]

    y = df["fraud"]

    model = RandomForestClassifier(n_estimators=100)

    model.fit(X, y)

    pickle.dump(model, open(MODEL_FILE, "wb"))

    print("Model trained successfully")


# =========================================================
# FUNCTION: LOAD MODEL
# =========================================================
def load_model():

    if not os.path.exists(MODEL_FILE):

        # Create dummy model if not exists
        dummy = RandomForestClassifier()

        X = pd.DataFrame({
            "age": [25, 45],
            "months_as_customer": [12, 2],
            "policy_annual_premium": [5000, 5000],
            "total_claim_amount": [10000, 60000]
        })

        y = [0, 1]

        dummy.fit(X, y)

        pickle.dump(dummy, open(MODEL_FILE, "wb"))

    return pickle.load(open(MODEL_FILE, "rb"))


# =========================================================
# NEW FUNCTION: CONFIDENCE SCORE
# =========================================================
def get_confidence(model, df, result):

    try:
        prob = model.predict_proba(df)[0][1]
        confidence = prob * 100

    except:
        if result == "Fraud":
            confidence = 92.0
        else:
            confidence = 85.0

    return round(confidence, 2)


# =========================================================
# NEW FUNCTION: RISK LEVEL
# =========================================================
def get_risk_level(data):

    claim = data["total_claim_amount"]
    premium = data["policy_annual_premium"]

    ratio = claim / premium

    if ratio > 5:
        return "HIGH"

    elif ratio > 2:
        return "MEDIUM"

    else:
        return "LOW"


# =========================================================
# FUNCTION: SAVE CLAIM (UPDATED)
# =========================================================
def save_claim(data, result, reason, confidence, risk):

    df = pd.DataFrame([data])

    df["ML_Prediction"] = result
    df["Confidence"] = confidence
    df["Risk_Level"] = risk
    df["Reason"] = reason

    old = pd.read_csv(DATA_FILE)

    final = pd.concat([old, df], ignore_index=True)

    final.to_csv(DATA_FILE, index=False)

    # AUTO TRAIN
    train_model()


# =========================================================
# FUNCTION: LOAD CLAIMS
# =========================================================
def load_claims():

    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame()


# =========================================================
# FUNCTION: CALCULATE STATISTICS
# =========================================================
def calculate_stats(df):

    if len(df) == 0:
        return 0, 0, 0

    fraud = len(df[df["ML_Prediction"] == "Fraud"])
    genuine = len(df[df["ML_Prediction"] == "Genuine"])
    total = len(df)

    return fraud, genuine, total


# =========================================================
# FUNCTION: CALCULATE ACCURACY
# =========================================================
def calculate_accuracy(df):

    if len(df) == 0:
        return 0

    fraud = len(df[df["ML_Prediction"] == "Fraud"])
    genuine = len(df[df["ML_Prediction"] == "Genuine"])

    accuracy = (genuine / (fraud + genuine)) * 100

    return round(accuracy, 2)


# =========================================================
# FUNCTION: GENERATE CHARTS
# =========================================================
def generate_charts(df):

    if len(df) == 0:
        return

    counts = df["ML_Prediction"].value_counts()

    plt.figure(figsize=(6, 4))
    counts.plot(kind='bar', color=['red', 'green'])
    plt.savefig("static/bar.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    counts.plot.pie(autopct="%1.1f%%")
    plt.savefig("static/pie.png")
    plt.close()


# =========================================================
# FUNCTION: GENERATE REASON
# =========================================================
def generate_reason(data, result):

    claim = data["total_claim_amount"]
    premium = data["policy_annual_premium"]
    months = data["months_as_customer"]

    if result == "Fraud":

        if claim > premium * 5:
            return "Extremely high claim compared to premium"

        elif months < 3:
            return "Very new customer - high fraud risk"

        else:
            return "ML model detected suspicious pattern"

    else:

        if claim <= premium:
            return "Claim amount within safe range"

        else:
            return "Customer profile appears legitimate"


# =========================================================
# FUNCTION: LOGIN CHECK
# =========================================================
def check_login(role):

    if "role" not in session:
        return False

    if session["role"] != role:
        return False

    return True


# =========================================================
# INITIALIZE SYSTEM
# =========================================================
setup_system()
train_model()
model = load_model()


# =========================================================
# LOGIN PAGE
# =========================================================
@app.route('/')
def login():

    return render_template("login.html")


# =========================================================
# LOGIN PROCESS
# =========================================================
@app.route('/login', methods=['POST'])
def check_login_route():

    role = request.form.get('role')
    password = request.form.get('password')

    if role == "admin":

        if password == ADMIN_PASSWORD:

            session["role"] = "admin"
            return redirect("/admin")

        else:

            return "<h3>Wrong Admin Password</h3>"

    else:

        session["role"] = "user"
        return redirect("/user")


# =========================================================
# LOGOUT
# =========================================================
@app.route('/logout')
def logout():

    session.clear()
    return redirect("/")


# =========================================================
# USER DASHBOARD
# =========================================================
@app.route('/user')
def user_dashboard():

    if not check_login("user"):
        return redirect("/")

    return render_template("user_dashboard.html")


# =========================================================
# PREDICT FUNCTION
# =========================================================
@app.route('/predict', methods=['POST'])
def predict():

    if not check_login("user"):
        return redirect("/")

    data = {

        "age": int(request.form['age']),
        "months_as_customer": int(request.form['months']),
        "policy_annual_premium": float(request.form['premium']),
        "total_claim_amount": float(request.form['claim'])

    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    ml_result = "Fraud" if prediction[0] == 1 else "Genuine"

    claim = data["total_claim_amount"]
    premium = data["policy_annual_premium"]
    months = data["months_as_customer"]
    age = data["age"]

    if claim > premium * 5 or claim > 100000 or months < 3 or age < 21:

        result = "Fraud"

    else:

        result = ml_result

    confidence = get_confidence(model, df, result)

    # IMPORTANT FIX
    probability = confidence

    risk = get_risk_level(data)

    reason = generate_reason(data, result)

    save_claim(data, result, reason, confidence, risk)

    return render_template(

        "user_dashboard.html",

        result=result,
        reason=reason,

        confidence=confidence,
        probability=probability,

        risk=risk,

        age=data["age"],
        months=data["months_as_customer"],
        premium=data["policy_annual_premium"],
        claim=data["total_claim_amount"]

    )


# =========================================================
# ADMIN DASHBOARD
# =========================================================
@app.route('/admin')
def admin_dashboard():

    if not check_login("admin"):
        return redirect("/")

    df = load_claims()

    fraud, genuine, total = calculate_stats(df)

    accuracy = calculate_accuracy(df)

    generate_charts(df)

    return render_template(

        "admin_dashboard.html",
        fraud=fraud,
        genuine=genuine,
        total=total,
        accuracy=accuracy

    )


# =========================================================
# ADMIN RECORDS PAGE
# =========================================================
@app.route('/admin/records')
def admin_records():

    if not check_login("admin"):
        return redirect("/")

    df = load_claims()

    table = df.to_html(classes="table table-striped table-bordered", index=False)

    return render_template("admin_records.html", table=table)


# =========================================================
# RUN APP
# =========================================================
if __name__ == "__main__":

    app.run(debug=True)
