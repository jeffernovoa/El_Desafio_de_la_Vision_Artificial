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
    top_classes = predict_class(st.session_state.model, image_array)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(uploaded_file, caption="Imagen subida", use_column_width=True)

    with col2:
        pred_class, pred_conf = top_classes[0]
        st.markdown(f"<h3 style='color:#2c3e50;'>Predicción principal: <span style='color:#3498db;'>{pred_class}</span></h3>", unsafe_allow_html=True)
        st.write(f"Confianza: **{pred_conf:.2%}**")
        st.progress(int(float(pred_conf) * 100))

        # Tabla de las 3 clases más probables
        st.subheader("📊 Top 3 predicciones")
        data = {
            "Clase": [c for c, _ in top_classes],
            "Confianza (%)": [f"{p*100:.2f}%" for _, p in top_classes],
            "Diferencia con la principal": [f"{(pred_conf - p)*100:.2f}%" if i != 0 else "-" for i, (_, p) in enumerate(top_classes)]
        }
        st.table(data)

        # Convertir a porcentaje y tipo int para la barra
        progress_value = int(float(pred_conf) * 100)
        st.progress(progress_value)


elif uploaded_file and "model" not in st.session_state:
    st.warning("⚠️ Entrena el modelo primero antes de hacer predicciones.")