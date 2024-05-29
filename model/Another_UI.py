import gradio as gr
import numpy as np
import pickle
# Loading the trained Random Forest model
Random_forest_model = pickle.load(open(
    "rfc_model.sav", 'rb'))

#loading scaler object
sc = pickle.load(open("standard_scaler", 'rb'))


# Define a function to predict using the model
def predict_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                    total_sulfur_dioxide, density, pH, sulphates, alcohol):
    # Create a numpy array from the input values
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                            pH, sulphates, alcohol]])
    print(input_data)
    # Use the model to predict the wine quality
    prediction = Random_forest_model.predict(sc.transform(input_data))

    # if prediction[0]==3:
    #     print(f"The quality is ={prediction[0]}")
    # elif prediction[0]==4:
    #     print(f"The quality is ={prediction[0]}")
    # elif prediction[0] == 5:
    #     print(f"The quality is ={prediction[0]}")
    # elif prediction[0] == 6:
    #     print(f"The quality is ={prediction[0]}")
    # elif prediction[0] == 7:
    #     print(f"The quality is ={prediction[0]}")
    # elif prediction[0] == 8:
    #     print(f"The quality is ={prediction[0]}")
    # elif prediction[0] == 9:
    #     print(f"The quality is ={prediction[0]}")
    quality_mapping = {3: "Low", 4: "Medium-Low", 5: "Medium", 6: "Medium-High", 7: "High", 8: "Very High",
                       9: "Extremely High"}
    return f"The quality is {quality_mapping[prediction[0]]}"



# Define the input sliders
fixed_acidity = gr.Slider(minimum=0, maximum=100, label="Fixed Acidity")
volatile_acidity = gr.Slider(minimum=0, maximum=100, label="Volatile Acidity")
citric_acid = gr.Slider(minimum=0, maximum=100, label="Citric Acid")
residual_sugar = gr.Slider(minimum=0, maximum=100, label="Residual Sugar")
chlorides = gr.Slider(minimum=0, maximum=100, label="Chlorides")
free_sulfur_dioxide = gr.Slider(minimum=0, maximum=100,label="Free Sulfur Dioxide")
total_sulfur_dioxide = gr.Slider(minimum=0, maximum=400, label="Total Sulfur Dioxide")
density = gr.Slider(minimum=0, maximum=100, label="Density")
pH = gr.Slider(minimum=0, maximum=100, label="pH")
sulphates = gr.Slider(minimum=0, maximum=100, label="Sulphates")
alcohol = gr.Slider(minimum=0, maximum=100, label="Alcohol")

# Define the interface
wine_quality_UI = gr.Interface(fn=predict_quality, inputs=[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                                           chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                                           density, pH, sulphates, alcohol], outputs="text",  title="Wine Quality Predictor")

# Launch the interface
wine_quality_UI.launch()
