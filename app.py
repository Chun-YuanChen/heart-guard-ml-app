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
        "ChestPainType": ChestPainType,
        "RestingBP": RestingBP,
        "Cholesterol": Cholesterol,
        "FastingBS": FastingBS,
        "RestingECG": RestingECG,
        "MaxHR": MaxHR,
        "ExerciseAngina": ExerciseAngina,
        "Oldpeak": Oldpeak,
        "ST_Slope": ST_Slope
    }])
    
    prediction = model.predict(input_df)[0]
    return "Heart Disease" if prediction == 1 else "Normal"

# Build interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Radio(["M", "F"], label="Sex"),
        gr.Radio(["TA", "ATA", "NAP", "ASY"], label="ChestPainType"),
        gr.Number(label="RestingBP"),
        gr.Number(label="Cholesterol"),
        gr.Radio([0, 1], label="FastingBS", type="value"),
        gr.Radio(["Normal", "ST", "LVH"], label="RestingECG"),
        gr.Number(label="MaxHR"),
        gr.Radio(["N", "Y"], label="ExerciseAngina"),
        gr.Number(label="Oldpeak"),
        gr.Radio(["Up", "Flat", "Down"], label="ST_Slope")
    ],
    outputs="text",
    title="Heart Disease Prediction",
    description="Enter patient data to predict heart disease risk."
)

iface.launch()