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
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to prevent the error

app = Flask(__name__)
app.config['DEBUG'] = True
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classIndex = json.load(open("class_indices.json"))



# MODELS = {
#     'best_model.h5': load_model('best_model.h5'),       # Carga el modelo CNN
#     'best_model2.joblib': load('best_model2.joblib'),   # Carga el modelo SVM
#     'best_model3.h5': load_model('best_model3.h5')      # Carga el modelo FFNN
# }

MODELS = {
    'best_model.h5': load_model('best_model.h5'),       # Carga el modelo CNN
    'best_model2.h5': load_model('best_model.h5'),       # Carga el modelo CNN
    'best_model3.h5': load_model('best_model.h5'),       # Carga el modelo CNN
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

            # Si se seleccionó la opción "TODOS", llama a graficas()
            if selected_model == 'ALL':
                graph_path = graficas(filename)
                return render_template('index.html', uploaded_image=filename, graph_image=graph_path)

            else:                
                current_model = MODELS[selected_model]  # Usamos el modelo ya cargado en memoria
                prediction, confidence = predict_mosquito_type(current_model, filename, selected_model)

                return render_template('index.html', uploaded_image=filename, prediction=prediction, confidence=confidence, selected_model=selected_model)
    
    return render_template('index.html', uploaded_image=None, prediction=None, confidence=None)

def predict_mosquito_type(model, img_path, model_name):
    img = Image.open(img_path).resize((100, 100))
    
    if model_name.endswith(".h5"):  # Si es un modelo Keras
        print("Modelo Keras")
        img = image.load_img(img_path, target_size=(100, 100))
        img_array = image.img_to_array(img) / 255.
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        percentage = (np.amax(prediction) * 100).round(2)
        predicted_index = np.argmax(prediction)

        labels = dict((v, k) for k, v in classIndex.items())  # flip the key, values in the dictionary
        predicted_label = labels[predicted_index]

    elif model_name.endswith(".joblib"):  # Si es el modelo SVM
        print("Modelo SVM")
        feature = hog(img, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

        svm_model = model["model"]
        le = model["labelencoder"]

        prediction = svm_model.predict([feature])
        predicted_label = le.inverse_transform(prediction)[0]

        # Obteniendo el porcentaje de confianza
        prediction_proba = svm_model.predict_proba([feature])
        percentage = (prediction_proba[0][prediction[0]] * 100).round(2)

    # Cierre la imagen después de usarla
    img.close()

    return predicted_label, percentage



def graficas(filename):
    model_names = list(MODELS.keys())
    accuracies = []
    mosquito_labels = []  # Almacenar los nombres de los mosquitos detectados
    confidences = []  # Almacenar la confianza de cada modelo

    # Hacer predicciones usando cada modelo
    for model_name in model_names:
        predicted_mosquito, accuracy = predict_mosquito_type(MODELS[model_name], filename, model_name)
        accuracies.append(accuracy)
        mosquito_labels.append(predicted_mosquito)
        confidences.append(accuracy)

    # Crear el directorio si no existe
    if not os.path.exists('static/graphs/'):
        os.makedirs('static/graphs/')

    # Definir colores para representar cada mosquito
    mosquito_colors = {
        'albopictus': '#7E7E7E',            # Gray
        'culex': '#2E8B57',                 # Sea Green
        'culiseta': '#CD5C5C',              # Indian Red
        'japonicus/koreicus': '#9370DB',    # Medium Purple
        'anopheles': '#FFA07A',             # Light Salmon
        'aegypti': '#FFFF66'                # Yellow
    }

    # Crear gráfica
    plt.figure(figsize=(10, 5))
    bars = plt.bar(model_names, accuracies)

    # Cambiar las etiquetas en el eje x
    new_labels = ['CNN', 'SVM', 'FFNN']
    plt.xticks()
    plt.gca().set_xticklabels(new_labels)

    # Agregar etiquetas de mosquito y confianza a las barras
    for i, bar in enumerate(bars):
        mosquito_label = mosquito_labels[i]
        confidence = confidences[i]
        bar.set_color(mosquito_colors.get(mosquito_label, 'gray'))
        plt.text(
            i,
            accuracy + 5,
            f'{mosquito_label}\n{confidence}%',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    plt.xlabel('Modelo')
    plt.ylabel('Accuracy (%)')
    plt.title('Comparativa de precisión entre modelos')
    plt.ylim(0, 100)  # para que el eje y vaya de 0 a 100

    # Guardar gráfica como imagen
    graph_path = os.path.join('static', 'graphs', 'model_comparison.png')
    plt.savefig(graph_path)
    plt.close()

    return graph_path


if __name__ == '__main__':
    app.run(debug=True)
