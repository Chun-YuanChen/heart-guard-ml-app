import gradio as gr
import pandas as pd
import joblib

# Load model
model = joblib.load("Random_Forest_best_model.pkl")

# Define prediction function
def predict(Age, Sex, ChestPainType, RestingBP, Cholesterol,
            FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    
    input_df = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "Type of chest pain": ChestPainType,
        "Resting blood pressure": RestingBP,
        "Serum cholesterol level": Cholesterol,
        "Fasting blood sugar status": FastingBS,
        "Resting electrocardiogram findings": RestingECG,
        "Maximum heart rate achieved": MaxHR,
        "Presence of exercise-induced angina": ExerciseAngina,
        "ST depression induced by exercise relative to rest": Oldpeak,
        "Slope of the peak exercise ST segment": ST_Slope
    }])
    
    prediction = model.predict(input_df)[0]
    return "Heart Disease" if prediction == 1 else "Normal"

# Build interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["M", "F"], label="Sex", type="index"),
        gr.Radio(["TA", "ATA", "NAP", "ASY"], label="Type of chest pain", type="index"),
        gr.Number(label="Resting blood pressure"),
        gr.Number(label="Serum cholesterol level"),
        gr.Radio(["0", "1"], label="Fasting blood sugar status", type="index"),
        gr.Radio(["Normal", "ST", "LVH"], label="Resting electrocardiogram findings", type="index"),
        gr.Number(label="Maximum heart rate achieved"),
        gr.Radio(["N", "Y"], label="Presence of exercise-induced angina", type="index"),
        gr.Number(label="ST depression induced by exercise relative to rest"),
        gr.Radio(["Up", "Flat", "Down"], label="Slope of the peak exercise ST segment", type="index")
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Enter patient data to predict heart disease risk."
)

iface.launch()