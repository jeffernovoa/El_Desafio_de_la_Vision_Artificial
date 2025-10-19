import matplotlib.pyplot as plt

def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Precisión en el conjunto de prueba: {test_acc:.2f}')

def plot_history(history):
    plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión validación')
    plt.xlabel('Épocas'); plt.ylabel('Precisión')
    plt.legend(); plt.show()

    plt.plot(history.history['loss'], label='Pérdida entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida validación')
    plt.xlabel('Épocas'); plt.ylabel('Pérdida')
    plt.legend(); plt.show()
