from flask import Flask, render_template, request, redirect
import os
from keras.preprocessing import image
import numpy as np
import pandas as pd
import json
from keras.models import load_model
from flask_bootstrap import Bootstrap
from joblib import load  # Importa la función load de joblib
from skimage.feature import hog
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend to prevent the error

app = Flask(__name__)
app.config['DEBUG'] = True
bootstrap = Bootstrap(app)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

classIndex = json.load(open("class_indices.json"))



# MODELS = {
#     'CNN': load_model('best_model.h5'),       # Carga el modelo CNN
#     'SVM': load('best_model2.joblib'),   # Carga el modelo SVM
#     'FFNN': load_model('best_model3.h5')      # Carga el modelo FFNN
# }

MODELS = {
    'CNN': load_model('best_model.h5'),  # Carga el modelo CNN
    'SVM': load_model('best_model.h5'),  # Carga el modelo SVM
    'FFNN': load_model('best_model.h5'),  # Carga el modelo FFNN
}

@app.route('/', methods=['GET', 'POST'])
def index():
    uploaded_image = None
    graph_path = None
    results = {}

    if request.method == 'POST':
        selected_model = request.form.get('model_selector', 'CNN')

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
                results['ALL'] = {'graph_image': graph_path,}

                for model_name in MODELS:
                    current_model = MODELS[model_name]
                    prediction, confidence = predict_mosquito_type(current_model, filename, model_name)
                    results[model_name] = {
                        'prediction': prediction,
                        'confidence': confidence,
                    }

            else:
                current_model = MODELS[selected_model]  # Usamos el modelo ya cargado en memoria
                prediction, confidence = predict_mosquito_type(current_model, filename, selected_model)
                results[selected_model] = {
                    'prediction': prediction,
                    'confidence': confidence,
                }
                graph_path = graficas(filename)

            zancudo_grafica()
            return render_template('index.html', uploaded_image=filename, results=results, selected_model=selected_model, graph_image=graph_path)

    return render_template('index.html', uploaded_image=None, prediction=None, confidence=None)

def predict_mosquito_type(model, img_path, model_name):
    img = Image.open(img_path).resize((100, 100))
    
    if model_name == "CNN" or model_name == "FFNN" or model_name == "SVM":  # Si es el modelo Keras
        print("Modelo Keras")
        img = image.load_img(img_path, target_size=(100, 100))
        img_array = image.img_to_array(img) / 255.
        img_batch = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_batch)
        percentage = (np.amax(prediction) * 100).round(2)
        predicted_index = np.argmax(prediction)

        labels = dict((v, k) for k, v in classIndex.items())  # flip the key, values in the dictionary
        predicted_label = labels[predicted_index]

    elif model_name == "SVM": # Si es el modelo SVM
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
    mosquito_labels = []  # Almacenar los nombres de los mosquitos detectados
    confidences = []  # Almacenar la confianza de cada modelo

    # Hacer predicciones usando cada modelo
    for model_name in model_names:
        predicted_mosquito, accuracy = predict_mosquito_type(MODELS[model_name], filename, model_name)
        mosquito_labels.append(predicted_mosquito)
        confidences.append(accuracy)

    # Crear un DataFrame para los datos
    data = {
        'Modelo': model_names,
        'Mosquito detectado': mosquito_labels,
        'Confianza (%)': confidences
    }
    df = pd.DataFrame(data)

    # Crear una figura interactiva con Plotly Express
    fig = px.bar(
        df,
        x='Modelo',
        y='Confianza (%)',
        color='Mosquito detectado',
        text='Confianza (%)',
        title='Comparativa de Clasificación de Modelos'
    )

    # Renderiza la figura como un archivo HTML
    fig.update_layout(autosize=False, width=600, height=400)
    graph_path = os.path.join('static', 'graphs', 'model_comparison.html')
    fig.write_html(graph_path)

    return graph_path


def zancudo_grafica():

    data = {
        'Categoría de Zancudo': ['albopictus', 'culex', 'culiseta', 'japonicus/koreicus', 'anopheles', 'aegypti'],
        'Enfermedades Transmitidas': [
            ['Chikungunya', 'Dengue', 'Zika', 'Fiebre Amarilla', 'Dirofilariasis'],
            ['Virus del Nilo Occidental', 'Encefalitis'],
            ['Encefalitis'],
            ['Virus del Nilo Occidental', 'Encefalitis', 'Dirofilariasis'],
            ['Malaria', 'Filariasis Linfática'],
            ['Dengue', 'Zika', 'Chikungunya', 'Fiebre Amarilla']
        ]
    }


    df = pd.DataFrame(data)

    # Crear un DataFrame adicional para expandir las categorías y enfermedades en filas separadas
    expanded_df = df.explode('Enfermedades Transmitidas')

    # Crear el gráfico de barras apiladas
    fig = px.bar(
        expanded_df,
        x='Categoría de Zancudo',
        y=expanded_df['Enfermedades Transmitidas'].apply(lambda x: 1),
        text='Enfermedades Transmitidas',
        title='Enfermedades Transmitidas por Categoría de Zancudo',
        category_orders={"Categoría de Zancudo": ["albopictus", "culex", "culiseta", "japonicus/koreicus", "anopheles", "aegypti"]}
    )

    # Configurar la apariencia del gráfico
    fig.update_layout(
        autosize=False,
        width=800,
        height=400,
        xaxis_title="Categoría de Zancudo",
        yaxis_title="Enfermedades Transmitidas"
    )

    # Renderiza la figura como un archivo HTML
    fig.update_layout(autosize=False, width=1000, height=400)
    graph_path = os.path.join('static', 'graphs', 'zancudos.html')
    fig.write_html(graph_path)


if __name__ == '__main__':
    app.run(debug=True)
