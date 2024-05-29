import dash
from dash import html, dcc, Input, Output
from dash.exceptions import PreventUpdate
import pickle

# Loading the trained Random Forest model
Random_forest_model = pickle.load(open(
    "rfc_model.sav", 'rb'))

#loading scaler object
sc = pickle.load(open("standard_scaler", 'rb'))

# Define the app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Wine Quality Predictor"),
    html.Div([
        dcc.Input(id='fixed_acidity', type='number', placeholder="Fixed Acidity"),
        dcc.Input(id='volatile_acidity', type='number', placeholder="Volatile Acidity"),
        dcc.Input(id='citric_acid', type='number', placeholder="Citric Acid"),
        dcc.Input(id='residual_sugar', type='number', placeholder="Residual Sugar"),
        dcc.Input(id='chlorides', type='number', placeholder="Chlorides"),
        dcc.Input(id='free_sulfur_dioxide', type='number', placeholder="Free Sulfur Dioxide"),
        dcc.Input(id='total_sulfur_dioxide', type='number', placeholder="Total Sulfur Dioxide"),
        dcc.Input(id='density', type='number', placeholder="Density"),
        dcc.Input(id='pH', type='number', placeholder="pH"),
        dcc.Input(id='sulphates', type='number', placeholder="Sulphates"),
        dcc.Input(id='alcohol', type='number', placeholder="Alcohol"),
        html.Button('Predict', id='predict_btn', n_clicks=0)
    ], style={'columnCount': 2}),
    html.H3("Predicted Wine Quality:"),
    html.Div(id='emoji_output')
])


# Define the callback
@app.callback(
    Output('emoji_output', 'children'),
    [Input('predict_btn', 'n_clicks')],
    [
        Input('fixed_acidity', 'value'),
        Input('volatile_acidity', 'value'),
        Input('citric_acid', 'value'),
        Input('residual_sugar', 'value'),
        Input('chlorides', 'value'),
        Input('free_sulfur_dioxide', 'value'),
        Input('total_sulfur_dioxide', 'value'),
        Input('density', 'value'),
        Input('pH', 'value'),
        Input('sulphates', 'value'),
        Input('alcohol', 'value')
    ]
)
def predict_wine_quality(n_clicks, fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                         chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH,
                         sulphates, alcohol):
    if n_clicks == 0:
        raise PreventUpdate

    # Collect user inputs
    new_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                 chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH,
                 sulphates, alcohol]]

    # Make the prediction using the Random Forest model
    prediction = Random_forest_model.predict(sc.transform(new_data))
    print(prediction)

    # Determine the emoji based on the predicted quality
    emoji = "😄" if prediction == 1 else "😞"

    return f"Predicted Quality: {emoji}"


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
