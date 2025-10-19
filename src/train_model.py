def train_model(model, x_train, y_train):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=64,
                        validation_split=0.1)
    return history
