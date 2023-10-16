from flask import Flask, render_template, request, redirect
import os
from keras.preprocessing import image
import numpy as np
import json
from keras.models import load_model
from flask_bootstrap import Bootstrap

app = Flask(__name__)
app.config['DEBUG'] = True
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('best_model.h5')
classIndex = json.load(open("class_indices.json"))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Comprueba si hay un archivo en la petición
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # Si el usuario no selecciona un archivo, el navegador podría enviar un archivo vacío.
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            # Guarda el archivo
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            # Llamada a la función de predicción
            prediction, confidence = predict_mosquito_type(model, filename)
            return render_template('index.html', uploaded_image=filename, prediction=prediction, confidence=confidence)
    
    # Si no es un POST o no se cumplen las condiciones anteriores, siempre se retorna esta línea
    return render_template('index.html', uploaded_image=None, prediction=None, confidence=None)

def predict_mosquito_type(model, img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_array = image.img_to_array(img) / 255.
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    percentage = np.amax(prediction) * 100
    predicted_index = np.argmax(prediction)

    labels = dict((v, k) for k, v in classIndex.items())  # flip the key, values in the dictionary
    predicted_label = labels[predicted_index]

    return predicted_label, percentage

if __name__ == '__main__':
    app.run(debug=True)
