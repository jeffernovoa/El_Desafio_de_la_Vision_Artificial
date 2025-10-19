# 🧠 Desafío de la Visión Artificial

Aplicación web interactiva desarrollada con **Streamlit** para entrenar y visualizar una red neuronal convolucional (**CNN**) usando el dataset **CIFAR-10**. Permite clasificar imágenes, ver métricas de entrenamiento, y explorar cómo el modelo interpreta diferentes clases.

---

## 🚀 Funcionalidades

- Entrenamiento de una CNN sobre CIFAR-10
- Visualización de precisión y pérdida por época
- Clasificación de imágenes subidas por el usuario
- Tabla con las 3 clases más probables y sus porcentajes
- Barra de progreso con confianza de la predicción
- Historial de imágenes clasificadas con selector para alternar entre ellas

---

## 🏗️ Estructura del proyecto

El_Desafio_de_la_Vision_Artificial/
│ 
 ├── venv/ # Entorno virtual 
 ├── src/ # Módulos de código
│ ├── load_data.py # Carga y preprocesamiento de CIFAR-10
│ ├── build_model.py # Arquitectura de la CNN
│ ├── train_model.py # Entrenamiento del modelo
│ ├── evaluate_model.py # Evaluación del modelo
│ ├── predict_image.py # Clasificación de imágenes subidas
│ 
 ├── app.py  # Interfaz web con Streamlit
 ├── requirements.txt  # Dependencias del proyecto
 ├── README.md

---

## 🧪 Requisitos

- Python 3.10
- pip

Instala las dependencias con:

``pip install -r requirements.txt``

## 📦 Dependencias principales

tensorflow==2.15.0
matplotlib==3.8.0
numpy==1.26.0
streamlit==1.29.0
pillow

---

## ▶️ Cómo ejecutar la app

``streamlit run app.py``

---

## 🔗 Link repositorio
https://github.com/jeffernovoa/El_Desafio_de_la_Vision_Artificial.git
