import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- LOOK AND FEEL ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800;900&display=swap');
html, body, [class*="css"] {
    font-family: 'Nunito Sans', sans-serif;
    font-weight: 700;
}
.stApp {
    background:
      radial-gradient(circle at 8% 8%, rgba(54, 190, 177, .22), transparent 28%),
      radial-gradient(circle at 92% 18%, rgba(94, 133, 230, .22), transparent 30%),
      linear-gradient(135deg, #101d35 0%, #172c4b 48%, #123d4b 100%);
    color: #f7fbff;
}
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
h1, h2, h3, p, label, .stMarkdown { color: #f7fbff; }
h1 {font-weight: 900 !important; letter-spacing: -1.5px;}
h2, h3 {font-weight: 900 !important;}
div[data-testid="stMetric"] {
    background: rgba(255,255,255,.10);
    border: 1px solid rgba(255,255,255,.18);
    padding: 18px 20px; border-radius: 18px;
}
div[data-testid="stMetric"] label {color:#c6d9ed !important;font-weight:800;}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#78e0d0;font-weight:900;}
div[data-testid="stForm"], div[data-testid="stExpander"], .stAlert {
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(255,255,255,.18);
    border-radius: 18px;
}
section[data-testid="stSidebar"] {background: #101a2d;}
section[data-testid="stSidebar"] * {font-weight:800;}
.stButton button, .stDownloadButton button {
    background: #53cbbb; color:#10233b; border:0; border-radius:12px;
    font-weight:900; padding:.55rem 1rem;
}
.stButton button:hover, .stDownloadButton button:hover {background:#8ce7d9;color:#10233b;}
div[data-baseweb="select"] > div, .stNumberInput input {
    background:#f6fbff; color:#14243c; border-radius:10px; font-weight:800;
}
hr {border-color:rgba(255,255,255,.18);}
.small-muted {color:#c6d9ed;font-size: .95rem;font-weight:700;}
.hero {
    background: linear-gradient(115deg, rgba(44,190,172,.22), rgba(97,126,220,.22));
    border:1px solid rgba(255,255,255,.20); border-radius:24px;
    padding:28px 30px; margin-bottom:22px;
}
.pill {display:inline-block;background:#d8fff5;color:#12384a;border-radius:30px;padding:6px 12px;font-weight:900;font-size:12px;}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
st.sidebar.title("🎓 Student Score Lab")
st.sidebar.caption("A simple machine-learning project")
uploaded = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])

@st.cache_data
def load_default():
    return pd.read_csv("StudentPerformanceFactors.csv")

try:
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        source_name = uploaded.name
    else:
        data = load_default()
        source_name = "StudentPerformanceFactors.csv"
except Exception:
    st.error("I couldn't find the CSV file. Keep StudentPerformanceFactors.csv in the same folder as this Python app, or upload it in the sidebar.")
    st.stop()

needed = ["Hours_Studied", "Sleep_Hours", "Exam_Score"]
if not all(c in data.columns for c in needed):
    st.error("This file needs these three columns: Hours_Studied, Sleep_Hours, Exam_Score.")
    st.write("Columns found:", list(data.columns))
    st.stop()

clean = data[needed].dropna().copy()
if len(clean) < 10:
    st.error("There are not enough complete rows to train a model.")
    st.stop()

# Train the model in the app so the numbers match the uploaded data.
X = clean[["Hours_Studied", "Sleep_Hours"]]
y = clean["Exam_Score"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
r2 = r2_score(y_test, test_predictions)
mae = mean_absolute_error(y_test, test_predictions)
mse = mean_squared_error(y_test, test_predictions)
rmse = np.sqrt(mse)

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
  <span class="pill">AI ACADEMY ASIA · JUNIOR AI ENGINEER</span>
  <h1 style="font-size:42px;margin:14px 0 8px;">Can study time and sleep help predict exam scores?</h1>
  <p class="small-muted">Explore student data, understand a simple AI model, and try your own prediction.</p>
</div>
""", unsafe_allow_html=True)

st.subheader("📌 Project at a glance")
a, b, c, d = st.columns(4)
a.metric("Student records", f"{len(clean):,}")
b.metric("Things we use", "2 inputs")
c.metric("What we predict", "Exam score")
d.metric("Test R² score", f"{r2:.3f}")

st.markdown("---")

# ---------- SIMPLE NAVIGATION ----------
page = st.sidebar.radio(
    "Choose a page",
    ["🏠 Overview", "📊 Explore the data", "🧮 Try a prediction", "🤖 Model results", "🎤 Presentation help"]
)

if page == "🏠 Overview":
    st.header("Welcome! 👋")
    st.write("This project asks whether the number of hours a student studies each week and sleeps each night can help estimate their exam score.")
    st.info("We use **Multiple Linear Regression**. It learns a simple mathematical relationship from examples in the dataset.")
    st.subheader("The three columns in our project")
    col1, col2, col3 = st.columns(3)
    col1.metric("Hours_Studied", "Study hours", "per week")
    col2.metric("Sleep_Hours", "Sleep hours", "per night")
    col3.metric("Exam_Score", "Exam score", "the value we predict")
    st.subheader("How it works")
    st.markdown("""
    1. **Load the data** – read student information from the CSV file.
    2. **Choose inputs** – use study hours and sleep hours.
    3. **Train the model** – let Linear Regression learn from 80% of the records.
    4. **Test the model** – check its predictions on the remaining 20%.
    5. **Try it yourself** – enter study and sleep hours to get an estimate.
    """)
    st.caption(f"Dataset file: {source_name}")

elif page == "📊 Explore the data":
    st.header("📊 Explore the student data")
    st.write("Each dot in the charts represents one student. Look for patterns, not proof of cause and effect.")
    c1, c2 = st.columns(2)
    with c1:
        fig1 = px.scatter(clean, x="Hours_Studied", y="Exam_Score",
                          title="Study hours and exam score",
                          labels={"Hours_Studied":"Study hours per week","Exam_Score":"Exam score"},
                          opacity=.55, color_discrete_sequence=["#55d6c2"])
        fig1.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        fig2 = px.scatter(clean, x="Sleep_Hours", y="Exam_Score",
                          title="Sleep hours and exam score",
                          labels={"Sleep_Hours":"Sleep hours per night","Exam_Score":"Exam score"},
                          opacity=.55, color_discrete_sequence=["#8caeff"])
        fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Average values")
    x1, x2, x3 = st.columns(3)
    x1.metric("Average study time", f"{clean.Hours_Studied.mean():.1f} hours/week")
    x2.metric("Average sleep", f"{clean.Sleep_Hours.mean():.1f} hours/night")
    x3.metric("Average exam score", f"{clean.Exam_Score.mean():.1f}")
    st.subheader("A quick look at the data")
    st.dataframe(clean.head(10), use_container_width=True)
    st.caption(f"Showing the first 10 of {len(clean):,} complete student records.")

elif page == "🧮 Try a prediction":
    st.header("🧮 Try the score predictor")
    st.write("Move the sliders or enter values. The model will estimate an exam score.")
    low_study = int(clean.Hours_Studied.min())
    high_study = int(clean.Hours_Studied.max())
    low_sleep = int(clean.Sleep_Hours.min())
    high_sleep = int(clean.Sleep_Hours.max())
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            study = st.slider("📚 Study hours per week", low_study, high_study,
                              int(round(clean.Hours_Studied.median())))
        with right:
            sleep = st.slider("🌙 Sleep hours per night", low_sleep, high_sleep,
                              int(round(clean.Sleep_Hours.median())))
        estimate = float(model.predict(pd.DataFrame({
            "Hours_Studied":[study], "Sleep_Hours":[sleep]
        }))[0])
        st.success(f"Estimated exam score: **{estimate:.1f}**")
        st.caption("This is a model estimate, not a guaranteed result or an official grade.")
    st.subheader("What did the model use?")
    st.write(f"- Study time: **{study} hours per week**")
    st.write(f"- Sleep time: **{sleep} hours per night**")
    st.write("- The model learned from the training part of the uploaded dataset.")

elif page == "🤖 Model results":
    st.header("🤖 How well did our model do?")
    st.write("We trained on 80% of the records and tested on 20% that the model had not seen during training.")
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Test R²", f"{r2:.3f}")
    q2.metric("MAE", f"{mae:.2f}")
    q3.metric("MSE", f"{mse:.2f}")
    q4.metric("RMSE", f"{rmse:.2f}")
    st.subheader("Understand the scores")
    st.markdown(f"""
    - **R² = {r2:.3f}:** the model explains about **{max(0,r2)*100:.1f}%** of the differences in exam scores in this test set. This is modest predictive power.
    - **MAE = {mae:.2f}:** predictions are off by about **{mae:.2f} score points on average**.
    - **MSE = {mse:.2f}:** the average squared prediction error. Larger errors receive extra penalty.
    - **RMSE = {rmse:.2f}:** a typical error size expressed in score points.
    """)
    st.subheader("The equation")
    st.code(
        f"Predicted score = {model.intercept_:.4f} "
        f"+ ({model.coef_[0]:.4f} × study hours) "
        f"+ ({model.coef_[1]:.4f} × sleep hours)",
        language="text"
    )
    st.warning("Important: this model uses only two factors. It does not prove that studying or sleeping more causes a higher or lower exam score. Other factors also matter.")
    st.write("Training and testing settings: 80% training data, 20% test data, random_state = 42.")

else:
    st.header("🎤 Get ready for your presentation")
    st.write("Your academy guide asks for a five-slide story. Use this simple outline:")
    slides = [
        ("1. The Question", "Can study time and sleep help predict exam scores? Why is this question interesting?"),
        ("2. The Data", f"We used {len(clean):,} student records and focused on study hours, sleep hours and exam scores."),
        ("3. The Model", "We chose Multiple Linear Regression because we are predicting a number using two input variables."),
        ("4. The Results", f"Our test R² is {r2:.3f} and our MAE is {mae:.2f}. Explain that the model has modest predictive power."),
        ("5. Demo & Takeaway", "Try the prediction sliders. Explain what you learned and what you would improve next.")
    ]
    for title, body in slides:
        with st.expander(title, expanded=True):
            st.write(body)
    st.subheader("Questions your teacher may ask")
    qa = [
        ("What is Linear Regression?", "A method that learns a mathematical relationship between input values and a number we want to predict."),
        ("Why do we split the data?", "To test the model on student records it did not use while learning."),
        ("What does MAE mean?", "The average size of the model's prediction errors, measured in score points."),
        ("Does this prove sleep changes exam scores?", "No. The model finds patterns in data; it cannot prove cause and effect.")
    ]
    for question, answer in qa:
        with st.expander(question):
            st.write(answer)

st.markdown("---")
st.caption("Student Score Lab · Data: StudentPerformanceFactors.csv · Educational project only")
