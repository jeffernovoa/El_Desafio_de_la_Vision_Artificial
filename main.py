from src.load_data import load_and_preprocess_data
from src.build_model import build_cnn_model
from src.train_model import train_model
from src.evaluate_model import evaluate_model, plot_history

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_cnn_model()
    history = train_model(model, x_train, y_train)
    evaluate_model(model, x_test, y_test)
    plot_history(history)

if __name__ == "__main__":
    main()
