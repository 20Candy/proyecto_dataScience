from flask import Flask, render_template, request, redirect, url_for
import os
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from modelo import predict_mosquito_type

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('best_model.h5')

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
            prediction = predict_mosquito_type(model, filename)
            return f"Predicción: {prediction}"
    return render_template('index.html')  


if __name__ == '__main__':
    app.run(debug=True)
