STUDENT SCORE LAB — PYTHON WEBSITE

1. Install Python on your computer if it is not already installed.
2. Open Command Prompt or Terminal in this folder.
3. Install the packages:
   pip install -r requirements.txt
4. Start the website:
   streamlit run app.py
5. Your browser should open the website automatically.

Keep app.py and StudentPerformanceFactors.csv in the same folder.
You can also upload a CSV from the sidebar inside the website.

The app trains its Linear Regression model from the CSV you load.
It uses Hours_Studied and Sleep_Hours to predict Exam_Score.
