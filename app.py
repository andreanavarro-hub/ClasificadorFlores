import streamlit as st
import joblib
import numpy as np

# Cargar el modelo y el codificador
mlp_model = joblib.load('/content/mlp_model.joblib')
label_encoder = joblib.load('/content/label_encoder.joblib')

st.title('Predicción de Especies de Flores')
st.write('Introduce las características de la flor para predecir su especie.')

# Cuadros de entrada para las características
sepal_length = st.number_input('Longitud del Sépalo (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
sepal_width = st.number_input('Ancho del Sépalo (cm)', min_value=0.0, max_value=10.0, value=0.3, step=0.1)
petal_length = st.number_input('Longitud del Pétalo (cm)', min_value=0.0, max_value=10.0, value=1.5, step=0.1)
petal_width = st.number_input('Ancho del Pétalo (cm)', min_value=0.0, max_value=10.0, value=1.0, step=0.1)

if st.button('Predecir Especie'):
    # Crear el array de características
    flower_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Realizar la predicción
    predicted_species_encoded = mlp_model.predict(flower_features)

    # Decodificar la predicción
    predicted_species_decoded = label_encoder.inverse_transform(predicted_species_encoded)

    st.success(f'La especie de la flor predicha es: {predicted_species_decoded[0]}')
