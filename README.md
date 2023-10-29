# Proyecto Data Science


Los modelos de aprendizaje automático entrenados para este proyecto son demasiado grandes para ser alojados en este repositorio de GitHub. Puedes descargar los modelos necesarios desde el siguiente enlace:

- [Modelo 1 (nombre del modelo)](enlace de descarga 1) - (Descripción breve)
- [Modelo 2 (nombre del modelo)](enlace de descarga 2) - (Descripción breve)
- [Modelo 3 (nombre del modelo)](enlace de descarga 3) - (Descripción breve)

**Nota:** Para descargar los modelos, debes tener una dirección de correo electrónico con el dominio uvg.edu.gt.

Una vez que hayas descargado los modelos, colócalos en la carpeta del proyecto siguiendo la estructura adecuada.

-----------------

El Ministerio de Salud y Asistencia Social (MSPAS) ha informado que Guatemala ha experimentado un aumento significativo en el número de casos de dengue en 2023, con más de 10,000 casos confirmados y 21 muertes hasta el 12 de agosto. El MSPAS ha declarado una alerta epidemiológica en todo el país para combatir la propagación del dengue (Gobierno De Guatemala, 2023).  

## Objetivo

Con esto en mente, el presente proyecto busca utilizar técnicas de procesamiento de imágenes y aprendizaje automático para identificar la especie de mosquito Aedes aegypti a partir de imágenes, con el fin de apoyar los esfuerzos de prevención del dengue en Guatemala.

## Integrantes
- Luis Santos
- Carol Arevalo
- Stefano Aragoni
- Diego Perdomo

## Librerias utilizadas
- pandas
- numpy
- matplotlib
- seaborn
- sklearn
- cv2
- PIL
- prettytable

## Variables en el dataset
1. <font color='orange'> **img_fName** </font>
    - Esta columna contiene el nombre de archivo de una imagen
    - Categórica
2. <font color='orange'> **img_w** </font>
    - Representa el ancho de la imagen en píxeles
    - Cuantitativa Discreta
3. <font color='orange'> **img_h** </font>
    - Representa la altura de la imagen en píxeles
    - Cuantitativa Discreta
4. <font color='orange'> **bbx_xtl** </font>
    - Esto es la coordenada x del punto superior izquierdo del cuadro delimitador (bounding box) alrededor de un objeto en la imagen
    - Cuantitativa Discreta
5. <font color='orange'> **bbx_ytl** </font>
    - Esto es la coordenada y del punto superior izquierdo del cuadro delimitador alrededor de un objeto en la imagen
    - Cuantitativa Discreta
6. <font color='orange'> **bbx_xbr** </font>
    - Esto es la coordenada x del punto inferior derecho del cuadro delimitador alrededor de un objeto en la imagen
    - Cuantitativa Discreta
7. <font color='orange'> **bbx_ybr** </font>
    - Esto es la coordenada y del punto inferior derecho del cuadro delimitador alrededor de un objeto en la imagen
    - Cuantitativa Discreta
8. <font color='orange'> **class_label** </font>
    - Esta columna contiene la etiqueta de categoría de mosquito en la imagen
    - Categórica