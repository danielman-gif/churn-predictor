import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import streamlit.components.v1 as components

# === ✅ Load env ===
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# === ✅ Load model ===
model = joblib.load("lightgbm_churn_model.pkl")

# === ✅ Title ===
st.title("🧩 E-Commerce Customer Churn Prediction")
st.markdown("""
This app predicts whether a customer is likely to churn using your trained churn model.
It also shows the SHAP explanation for each prediction — and you can ask an AI chatbot about it!
""")

# === ✅ Init LLM ===
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=google_api_key
)

# === ✅ Inputs ===
st.header("🔍 Customer Features")

recency = st.slider("Recency (days since last purchase)", min_value=0, max_value=500, value=90)
frequency = st.number_input("Frequency (number of orders)", min_value=1, value=5)
monetary = st.number_input("Monetary (total spend)", min_value=1, value=500)

country = st.selectbox("Country", [
    "United Kingdom", "Germany", "France", "Netherlands", "Other"
])

# === ✅ Feature prep ===
frequency_log = np.log1p(frequency)
monetary_log = np.log1p(monetary)

country_columns = [
    "Country_United Kingdom",
    "Country_Germany",
    "Country_France",
    "Country_Netherlands",
    "Country_Other"
]

country_data = [0] * len(country_columns)
if country in ["United Kingdom", "Germany", "France", "Netherlands"]:
    idx = country_columns.index(f"Country_{country}")
else:
    idx = country_columns.index("Country_Other")
country_data[idx] = 1

input_data = pd.DataFrame([[recency, frequency_log, monetary_log]], columns=[
    "Recency", "Frequency_log", "Monetary_log"
])

# === ✅ Predict button ===
run_predict = st.button("Predict Churn", key="predict_button")

if run_predict:
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)[0][1]

    st.session_state["latest_prediction"] = int(prediction[0])
    st.session_state["latest_proba"] = float(proba)
    st.session_state["latest_input"] = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "Country": country
    }

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    if isinstance(shap_values, list):
        shap_value = shap_values[1]
    else:
        shap_value = shap_values

    st.session_state["latest_shap"] = shap_value.tolist()
    st.session_state["shap_valid"] = True  # Sticky flag for rerun

    st.write(f"**Churn Prediction:** {'Churned' if prediction[0]==1 else 'Active'}")
    st.write(f"**Probability of Churn:** {proba:.2%}")

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    force_plot = shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_value,
        input_data
    )

    st.subheader("🔎 SHAP Explanation")
    st_shap(force_plot)

# === ✅ Sticky rerun block — runs only if user did NOT just click Predict ===
if st.session_state.get("shap_valid", False) and not run_predict:
    st.write(f"**Churn Prediction:** {'Churned' if st.session_state['latest_prediction'] == 1 else 'Active'}")
    st.write(f"**Probability of Churn:** {st.session_state['latest_proba']:.2%}")

    def st_shap(plot, height=None):
        shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
        components.html(shap_html, height=height)

    shap_value = np.array(st.session_state["latest_shap"])
    input_data_cached = pd.DataFrame([[recency, frequency_log, monetary_log]], columns=[
        "Recency", "Frequency_log", "Monetary_log"
    ])

    explainer = shap.TreeExplainer(model)
    force_plot = shap.force_plot(
        explainer.expected_value,
        shap_value,
        input_data_cached
    )

    st.subheader("🔎 SHAP Explanation")
    st_shap(force_plot)


# === Sidebar chatbot with real context ===
with st.sidebar:
    st.header("💬 Ask about this project or your prediction!")

    user_q = st.text_input("Ask me anything:")

    if user_q:
        prediction = st.session_state.get("latest_prediction", None)
        proba = st.session_state.get("latest_proba", None)
        shap_value = st.session_state.get("latest_shap", None)
        input_used = st.session_state.get("latest_input", None)

        if prediction is None:
            prediction_info = "No prediction has been made yet."
        else:
            shap_explanation = ""
            if shap_value:
                row = shap_value[0]
                recency_shap = round(row[0], 4)
                frequency_shap = round(row[1], 4)
                monetary_shap = round(row[2], 4)

                # Recency: negative is good (less churn)
                if recency_shap < 0:
                    recency_text = (
                        f"The SHAP value is {recency_shap}. "
                        f"This means that a higher recency (more days since last purchase) reduces churn probability, "
                        f"so recent buyers are less likely to churn."
                    )
                else:
                    recency_text = (
                        f"The SHAP value is {recency_shap}. "
                        f"This means that a higher recency increases churn probability, "
                        f"so less recent buyers are more likely to churn."
                    )

                # Frequency: positive pushes toward churn
                if frequency_shap > 0:
                    frequency_text = (
                        f"The SHAP value is {frequency_shap}. "
                        f"This means that higher frequency increases the churn probability — "
                        f"so for this customer, buying more often is linked to higher churn risk."
                    )
                else:
                    frequency_text = (
                        f"The SHAP value is {frequency_shap}. "
                        f"This means that higher frequency reduces the churn probability — "
                        f"so buying more often lowers churn risk."
                    )

                # Monetary: same
                if monetary_shap > 0:
                    monetary_text = (
                        f"The SHAP value is {monetary_shap}. "
                        f"This means that higher spend increases churn probability — "
                        f"so spending more is linked to slightly higher churn risk."
                    )
                else:
                    monetary_text = (
                        f"The SHAP value is {monetary_shap}. "
                        f"This means that higher spend reduces churn probability — "
                        f"so spending more lowers churn risk."
                    )

                shap_explanation = f"""
**Recency:** {recency_text}

**Frequency:** {frequency_text}

**Monetary:** {monetary_text}
"""

            prediction_info = f"""
Latest prediction:
- Churned: {'Yes' if prediction == 1 else 'No'}
- Probability: {proba:.2%}
Input:
{input_used}

SHAP Explanation:
{shap_explanation}
"""

        # ✅ Add your static project info
        project_info = """
Project details:
- Dataset: Online Retail II.
- Data cleaning: removed returns, bad debt, duplicates.
- Features: RFM (Recency, Frequency, Monetary) — log transformed.
- Models tested: Logistic Regression, Random Forest, XGBoost, LightGBM, KNN.
- Best: LightGBM (ROC-AUC ≈ 0.80) — used for this deployed version.
- Evaluation: Confusion Matrix, ROC-AUC, SHAP explanations.
- Built with: Python, Streamlit, LangChain chatbot, OpenAI LLM.
"""

        messages = [
            SystemMessage(
                content=f"""
You are a helpful assistant for explaining this churn prediction project.
Use this static project info plus the latest prediction context.

{project_info}

{prediction_info}

Answer clearly.
"""
            ),
            HumanMessage(content=user_q)
        ]

        response = llm.invoke(messages)
        st.write(response.content)


