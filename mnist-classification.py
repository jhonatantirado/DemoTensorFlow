import tensorflow as tf
mnist = tf.keras.datasets.mnist

"""
el dataset MNIST consiste en imagenes de numeros escritos a mano, de 28x28 pixeles, en escala de grises
son 60000 imagenes de entrenamiento, y 10000 de prueba
los numeros son del 0 al 9
"""
(x_train, y_train),(x_test, y_test) = mnist.load_data()
#normalizar la intesidad de los pixeles para obtener valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0
#si no se aplica normalizacion, los resultados pueden ser muy diferentes
#en este caso, la precision baja de 96 a 56%

"""
construir el modelo usando Keras, concatenando las capas de la red neuronal
la capa de entrada(Flatten) consiste en un vector de 784x1, ya que recibe de entrada una matriz de 28x28
la primera capa oculta(Dense) tiene 128 neuronas, usando la funcion de activacion RELU
RELU f(x)=max(0,x), devuelve CERO si X es negativo, o X si es positivo
la segunda capa oculta(Dropout), setea las entradas a CERO, aleatoriamente al 20% de neuronas
con el fin de evitar el overfitting
la capa final(Dense) tiene 10 neuronas ya que son 10 las clases a predecir, 
usando la funcion de activacion SOFTMAX ya que se calcula la probabilidad de cada clase 
y todo debe sumar el 100%
 f(x)= e^x/ sum(e^x)
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

"""
Optimizador "adam": adaptive moment estimation
se tiene y actualiza una taza de aprendizaje por cada peso de la red
a diferencia de la descenso de gradiente estocástica que usa una taza de aprendizaje global
popular en computer vision
Loss function "sparse_categorical_crossentropy": es la función de costo que se quiere optimizar
en este caso,se quiere minimizar el costo, el error
funciones disponibles: mean_squared_error, mean_squared_logarithmic_error
categorical_crossentropy: devuelve 0 o 1
sparse_categorical_crossentropy: devuelve un número entero, no solo 0 o 1
se usa cuando se quiere obtener la distribucion de probabilidad 
"""

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

#opcional: convertir a 0s y 1s para usar categorical_crossentropy
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

#detener el entrenamiento si "val_loss" deja de mejorar en 2 epocas
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

"""
entrenar el modelo, en 5 epocas o pasadas
usar callbacks para extender el comportamiento durante el entrenamiento
los callbacks se activan cuando se usan 20 epocas, por ejemplo
otro callback registra logs para TensorBoard
Para ver los datos en TensorBoard
Ejecutar: tensorboard --logdir=path/to/log-directory
Ir a: http://localhost:6006
"""
num_epochs = 1
model.fit(x_train, y_train, epochs = num_epochs,callbacks=callbacks,
          validation_data=(x_test, y_test))

#validar el modelo
model.evaluate(x_test, y_test)

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Save entire model to a HDF5 file
model.save('my_entire_model.h5')

#Convertir modelo a formato TFLITE para usar en una app movil
converter = tf.lite.TFLiteConverter.from_keras_model_file("my_entire_model.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

"""
guardar el modelo en formato PNG
rankdir='TB' grafico vertical, "LR" horizontal
el numero que aparece al inicio del grafico es incorrecto, debido a un bug de Keras
deberia indicar la cantidad de parametros(weight, bias) del modelo
"""
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='LR')

#mostrar resumen del modelo
print(model.summary())

# Serialize a model to JSON format
json_string = model.to_json()
print (json_string)

#No disponible en TensorFlow 1.13
#tf.keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)