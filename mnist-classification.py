import tensorflow as tf
mnist = tf.keras.datasets.mnist

#el dataset MNIST consiste en imagenes de numeros escritos a mano, de 28x28 pixeles, en escala de grises
#son 60000 imagenes de entrenamiento, y 10000 de prueba
#los numeros son del 0 al 9
(x_train, y_train),(x_test, y_test) = mnist.load_data()
#normalizar la intesidad de los pixeles para obtener valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0
#si no se aplica normalizacion, los resultados pueden ser muy diferentes
#en este caso, la precision baja de 96 a 56%

#construir el modelo usando Keras, concatenando las capas de la red neuronal
#la capa de entrada(Flatten) consiste en un vector de 784x1, ya que recibe de entrada una matriz de 28x28
#la primera capa oculta(Dense) tiene 128 neuronas, usando la funcion de activacion RELU
#RELU f(x)=max(0,x), devuelve CERO si X es negativo, o X si es positivo
#la segunda capa oculta(Dropout), setea las entradas a CERO, aleatoriamente al 20% de neuronas
#con el fin de evitar el overfitting
#la capa final(Dense) tiene 10 neuronas ya que son 10 las clases a predecir, 
#usando la funcion de activacion SOFTMAX ya que se calcula la probabilidad de cada clase 
#y todo debe sumar el 100%
# f(x)= e^x/ sum(e^x)
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

#Optimizador "adam": adaptive moment estimation
#se tiene y actualiza una taza de aprendizaje por cada peso de la red
#a diferencia de la descenso de gradiente estocástica que usa una taza de aprendizaje global
#popular en computer vision
#Loss function "sparse_categorical_crossentropy": es la función de costo que se quiere optimizar
#en este caso,se quiere minimizar el costo, el error
#funciones disponibles: mean_squared_error, mean_squared_logarithmic_error
#categorical_crossentropy: devuelve 0 o 1
#sparse_categorical_crossentropy: devuelve un número entero, no solo 0 o 1
#se usa cuando se quiere obtener la distribucion de probabilidad 

#usar sparse_categorical_crossentropy
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#usar categorical_crossentropy
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#opcional: convertir a 0s y 1s para usar categorical_crossentropy
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)


#entrenar el modelo, en 5 epocas o pasadas
num_epochs = 1
model.fit(x_train, y_train, epochs = num_epochs)

#opcional: convertir a 0s y 1s para usar categorical_crossentropy
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#validar el modelo
model.evaluate(x_test, y_test)

#guardar el modelo en formato PNG
#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

#tf.keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)