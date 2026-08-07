import pickle
import numpy as np
import gradio as gr

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------
import os
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    # Try to help the user locate the file instead of just failing.
    search_dir = os.path.dirname(MODEL_PATH)
    candidates = glob.glob(os.path.join(SCRIPT_DIR, "**", "*.pkl"), recursive=True)
    msg = [
        f"Could not find 'model.pkl' at: {MODEL_PATH}",
        "",
        f"Files currently in {search_dir}:" if os.path.isdir(search_dir) else f"Folder does not exist: {search_dir}",
    ]
    if os.path.isdir(search_dir):
        for name in sorted(os.listdir(search_dir)):
            msg.append(f"  - {name}")
    if candidates:
        msg.append("")
        msg.append(f"Found a .pkl file with a different name: {candidates[0]}")
        msg.append("Rename it to 'model.pkl', or update MODEL_PATH above to point to it directly.")
    raise FileNotFoundError("\n".join(msg))

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------------------------------------
# Encoding maps
#
# The pickle file only contains the trained RandomForestClassifier - it does
# NOT include the LabelEncoders that were used on the categorical columns
# during training. This app assumes the standard preprocessing used for the
# well-known "Telco Customer Churn" dataset, where each categorical column
# was passed through sklearn's LabelEncoder (which assigns codes in
# alphabetical order of the category names). If your original training
# pipeline encoded things differently, predictions will not be accurate -
# swap the maps below to match your actual encoders.
# ---------------------------------------------------------------------------
BINARY_MAP = {"No": 0, "Yes": 1}
GENDER_MAP = {"Female": 0, "Male": 1}
TRIPLE_MAP = {"No": 0, "No internet service": 1, "Yes": 2}  # security/backup/etc.
MULTI_LINES_MAP = {"No": 0, "No phone service": 1, "Yes": 2}
INTERNET_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
PAYMENT_MAP = {
    "Bank transfer (automatic)": 0,
    "Credit card (automatic)": 1,
    "Electronic check": 2,
    "Mailed check": 3,
}

FEATURE_ORDER = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def predict_churn(
    gender, senior_citizen, partner, dependents, tenure,
    phone_service, multiple_lines, internet_service, online_security,
    online_backup, device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing, payment_method,
    monthly_charges, total_charges,
):
    row = {
        "gender": GENDER_MAP[gender],
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": BINARY_MAP[partner],
        "Dependents": BINARY_MAP[dependents],
        "tenure": tenure,
        "PhoneService": BINARY_MAP[phone_service],
        "MultipleLines": MULTI_LINES_MAP[multiple_lines],
        "InternetService": INTERNET_MAP[internet_service],
        "OnlineSecurity": TRIPLE_MAP[online_security],
        "OnlineBackup": TRIPLE_MAP[online_backup],
        "DeviceProtection": TRIPLE_MAP[device_protection],
        "TechSupport": TRIPLE_MAP[tech_support],
        "StreamingTV": TRIPLE_MAP[streaming_tv],
        "StreamingMovies": TRIPLE_MAP[streaming_movies],
        "Contract": CONTRACT_MAP[contract],
        "PaperlessBilling": BINARY_MAP[paperless_billing],
        "PaymentMethod": PAYMENT_MAP[payment_method],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    X = np.array([[row[f] for f in FEATURE_ORDER]])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    churn_prob = float(proba[1])
    stay_prob = float(proba[0])

    label = "⚠️ Likely to Churn" if pred == 1 else "✅ Likely to Stay"

    result_md = (
        f"### {label}\n\n"
        f"- **Churn probability:** {churn_prob:.1%}\n"
        f"- **Retention probability:** {stay_prob:.1%}\n"
    )

    return result_md, {"Churn": churn_prob, "No Churn": stay_prob}


with gr.Blocks(title="Telco Customer Churn Predictor") as demo:
    gr.Markdown(
        """
        # 📊 Telco Customer Churn Predictor
        Fill in the customer's details below to predict whether they are likely
        to churn, using a trained Random Forest model.

        > ⚠️ **Note:** The uploaded model file only contains the trained
        > classifier, not the original label encoders. This app encodes
        > categorical fields using the conventional alphabetical-order
        > encoding for the standard Telco Churn dataset. If your training
        > pipeline used different encoding, adjust the maps in `app.py`.
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Customer Profile")
            gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
            senior_citizen = gr.Radio(["No", "Yes"], value="No", label="Senior Citizen")
            partner = gr.Radio(["No", "Yes"], value="No", label="Has Partner")
            dependents = gr.Radio(["No", "Yes"], value="No", label="Has Dependents")
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")

        with gr.Column():
            gr.Markdown("### Services")
            phone_service = gr.Radio(["No", "Yes"], value="Yes", label="Phone Service")
            multiple_lines = gr.Radio(
                ["No", "No phone service", "Yes"], value="No", label="Multiple Lines"
            )
            internet_service = gr.Radio(
                ["DSL", "Fiber optic", "No"], value="DSL", label="Internet Service"
            )
            online_security = gr.Radio(
                ["No", "No internet service", "Yes"], value="No", label="Online Security"
            )
            online_backup = gr.Radio(
                ["No", "No internet service", "Yes"], value="No", label="Online Backup"
            )
            device_protection = gr.Radio(
                ["No", "No internet service", "Yes"], value="No", label="Device Protection"
            )
            tech_support = gr.Radio(
                ["No", "No internet service", "Yes"], value="No", label="Tech Support"
            )
            streaming_tv = gr.Radio(
                ["No", "No internet service", "Yes"], value="No", label="Streaming TV"
            )
            streaming_movies = gr.Radio(
                ["No", "No internet service", "Yes"], value="No", label="Streaming Movies"
            )

        with gr.Column():
            gr.Markdown("### Account & Billing")
            contract = gr.Radio(
                ["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract",
            )
            paperless_billing = gr.Radio(["No", "Yes"], value="Yes", label="Paperless Billing")
            payment_method = gr.Dropdown(
                [
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                    "Electronic check",
                    "Mailed check",
                ],
                value="Electronic check",
                label="Payment Method",
            )
            monthly_charges = gr.Number(value=70.0, label="Monthly Charges ($)")
            total_charges = gr.Number(value=840.0, label="Total Charges ($)")

    predict_btn = gr.Button("🔮 Predict Churn", variant="primary")

    with gr.Row():
        result_output = gr.Markdown(label="Prediction")
        prob_output = gr.Label(label="Probabilities")

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service, online_security,
            online_backup, device_protection, tech_support, streaming_tv,
            streaming_movies, contract, paperless_billing, payment_method,
            monthly_charges, total_charges,
        ],
        outputs=[result_output, prob_output],
    )

if __name__ == "__main__":
    demo.launch()