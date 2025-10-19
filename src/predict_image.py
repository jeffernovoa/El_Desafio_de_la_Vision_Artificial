import numpy as np
from PIL import Image

def preprocess_uploaded_image(uploaded_file):
    image = Image.open(uploaded_file).resize((32, 32))
    image_array = np.array(image).astype('float32') / 255.0
    if image_array.shape == (32, 32):
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_class(model, image_array):
    predictions = model.predict(image_array)
    class_index = np.argmax(predictions)
    class_names = ['Avión', 'Automóvil', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']
    return class_names[class_index], predictions[0][class_index]
