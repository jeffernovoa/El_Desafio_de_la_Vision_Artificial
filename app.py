import streamlit as st
from src.load_data import load_and_preprocess_data
from src.build_model import build_cnn_model
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.predict_image import preprocess_uploaded_image, predict_class
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Visión Artificial con CIFAR-10", layout="wide", page_icon="🧠")

# Estilo personalizado
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2c3e50;}
    .stButton>button {background-color: #3498db; color: white;}
    </style>
""", unsafe_allow_html=True)

# Encabezado
st.title("🧠 Desafío de la Visión Artificial")
st.markdown("Explora el poder de las redes neuronales convolucionales con el dataset CIFAR-10.")

# Barra lateral
st.sidebar.header("⚙️ Opciones")
train_now = st.sidebar.button("Entrenar modelo")
uploaded_file = st.sidebar.file_uploader("📷 Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

# Entrenamiento del modelo
if train_now:
    with st.spinner("🔄 Cargando y entrenando modelo..."):
        x_train, y_train, x_test, y_test = load_and_preprocess_data()
        model = build_cnn_model()
        history = train_model(model, x_train, y_train)
        test_loss, test_acc = evaluate_model(model, x_test, y_test)

        st.session_state.model = model
        st.session_state.history = history
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test

    st.success("✅ Modelo entrenado correctamente")

    # Visualización de métricas
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Precisión")
        fig_acc = plt.figure()
        plt.plot(history.history['accuracy'], label='Entrenamiento')
        plt.plot(history.history['val_accuracy'], label='Validación')
        plt.xlabel("Épocas"); plt.ylabel("Precisión"); plt.legend()
        st.pyplot(fig_acc)

    with col2:
        st.subheader("📉 Pérdida")
        fig_loss = plt.figure()
        plt.plot(history.history['loss'], label='Entrenamiento')
        plt.plot(history.history['val_loss'], label='Validación')
        plt.xlabel("Épocas"); plt.ylabel("Pérdida"); plt.legend()
        st.pyplot(fig_loss)

    st.metric(label="📈 Precisión en prueba", value=f"{test_acc:.2%}")

st.divider()

# Clasificación de imagen
if uploaded_file and "model" in st.session_state:
    st.subheader("🔍 Clasificación de imagen subida")
    image_array = preprocess_uploaded_image(uploaded_file)
    class_name, confidence = predict_class(st.session_state.model, image_array)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
    with col2:
        st.write(f"**Predicción:** `{class_name}`")
        st.progress(confidence)
        st.write(f"Confianza: **{confidence:.2%}**")

elif uploaded_file and "model" not in st.session_state:
    st.warning("⚠️ Entrena el modelo primero antes de hacer predicciones.")

