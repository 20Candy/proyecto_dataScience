from flask import Flask, render_template, request, redirect
import os
from keras.preprocessing import image
import numpy as np
import json
from keras.models import load_model
from flask_bootstrap import Bootstrap
from joblib import load  # Importa la función load de joblib
from skimage.feature import hog
from keras.preprocessing.image import img_to_array
from PIL import Image

app = Flask(__name__)
app.config['DEBUG'] = True
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classIndex = json.load(open("class_indices.json"))



MODELS = {
    'best_model.h5': load_model('best_model.h5'),
    'best_model2.joblib': load('best_model2.joblib'),  # Carga el modelo SVM
    'best_model3.h5': load_model('best_model3.h5')
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        selected_model = request.form.get('model_selector', 'best_model.h5')

        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename).replace("\\", "/")
            file.save(filename)
            
            current_model = MODELS[selected_model]  # Usamos el modelo ya cargado en memoria
            prediction, confidence = predict_mosquito_type(current_model, filename, selected_model)

            return render_template('index.html', uploaded_image=filename, prediction=prediction, confidence=confidence, selected_model=selected_model)
    
    return render_template('index.html', uploaded_image=None, prediction=None, confidence=None)

def predict_mosquito_type(model, img_path, model_name):
    img = Image.open(img_path).resize((100, 100))
    
    if model_name.endswith(".h5"):  # Si es un modelo Keras
        img_array = img_to_array(img) / 255.
        img_batch = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_batch)
        percentage = (np.amax(prediction) * 100).round(2)
        predicted_index = np.argmax(prediction)

        labels = dict((v, k) for k, v in classIndex.items())  # flip the key, values in the dictionary
        predicted_label = labels[predicted_index]

    elif model_name.endswith(".joblib"):  # Si es el modelo SVM
        feature = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

        svm_model = model["model"]
        le = model["labelencoder"]

        prediction = svm_model.predict([feature])
        predicted_label = le.inverse_transform(prediction)[0]

        # Obteniendo el porcentaje de confianza
        prediction_proba = svm_model.predict_proba([feature])
        percentage = (prediction_proba[0][prediction[0]] * 100).round(2)

    return predicted_label, percentage



if __name__ == '__main__':
    app.run(debug=True)
