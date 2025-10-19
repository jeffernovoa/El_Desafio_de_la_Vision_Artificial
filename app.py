import streamlit as st
from src.load_data import load_and_preprocess_data
from src.build_model import build_cnn_model
from src.train_model import train_model
from src.evaluate_model import evaluate_model, plot_history
import matplotlib.pyplot as plt

st.set_page_config(page_title="CNN CIFAR-10", layout="wide")

st.title("🌟 Desafío de la Visión Artificial con CIFAR-10")
st.markdown("Entrena una red neuronal convolucional y visualiza los resultados directamente en tu navegador.")

if st.button("Entrenar modelo"):
    with st.spinner("Cargando datos..."):
        x_train, y_train, x_test, y_test = load_and_preprocess_data()

    with st.spinner("Construyendo modelo..."):
        model = build_cnn_model()

    with st.spinner("Entrenando modelo..."):
        history = train_model(model, x_train, y_train)

    st.success("✅ Entrenamiento completado")

    st.subheader("📊 Precisión y pérdida")
    fig_acc, fig_loss = plt.figure(), plt.figure()

    plt.figure(fig_acc.number)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.xlabel("Épocas"); plt.ylabel("Precisión"); plt.legend()
    st.pyplot(fig_acc)

    plt.figure(fig_loss.number)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.xlabel("Épocas"); plt.ylabel("Pérdida"); plt.legend()
    st.pyplot(fig_loss)

    st.subheader("📈 Evaluación final")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    st.write(f"**Precisión en el conjunto de prueba:** {test_acc:.2%}")
