import tensorflow as tf
mnist = tf.keras.datasets.mnist

"""
El dataset MNIST consiste en imágenes de números escritos a mano, de 28x28 píxeles, en escala de grises
Son 60000 imágenes de entrenamiento, y 10000 de prueba
Los números son del 0 al 9, o sea, 10 clases
"""
(x_train, y_train),(x_test, y_test) = mnist.load_data()
#Normalizar la intesidad de los píxeles para obtener valores entre 0 y 1
x_train, x_test = x_train / 255.0, x_test / 255.0
#Si no se aplica normalización, los resultados pueden ser muy diferentes
#En este caso, la precisión baja de 96% a 56%

"""
Red Neuronal
Construir el modelo usando Keras, concatenando las capas de la red neuronal
La capa de entrada(Flatten) consiste en un vector de 784x1, ya que recibe de entrada una matriz de 28x28
La primera capa oculta(Dense) tiene 128 neuronas, usando la función de activación RELU
RELU f(x)=max(0,x), devuelve CERO si X es negativo, o X si es positivo
La segunda capa oculta(Dropout), setea las entradas a CERO, aleatoriamente al 20% de neuronas con el fin de evitar el overfitting (que la red memorice las respuestas en vez de aprender los patrones)
La capa final(Dense) tiene 10 neuronas ya que son 10 las clases a predecir, usando la función de activación SOFTMAX ya que se calcula la probabilidad de cada clase y todo debe sumar el 100%
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
Se tiene y actualiza una taza de aprendizaje por cada peso de la red, a diferencia de la descenso de gradiente estocástica (SGD) que usa una taza de aprendizaje global
Es popular en Computer Vision
Loss function "sparse_categorical_crossentropy": es la función de costo que se quiere optimizar
En este caso, se quiere minimizar el error
Funciones disponibles: mean_squared_error, mean_squared_logarithmic_error
"categorical_crossentropy": devuelve 0 o 1
"sparse_categorical_crossentropy": devuelve un número entero, no solo 0 o 1
Se usa cuando se quiere obtener la distribución de probabilidad 
"""

#Usar sparse_categorical_crossentropy
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Usar categorical_crossentropy
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])

#Opcional: convertir a 0s y 1s para usar categorical_crossentropy
#y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

#Opcional: convertir a 0s y 1s para usar categorical_crossentropy
#y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

"""
Uso de "callbacks"
Detener el entrenamiento si "val_loss" (error en la etapa de validación) deja de mejorar (disminuir) en 2 épocas consecutivas
Capturar logs para TensorBoard en la carpeta ./logs para su posterior visualización
"""
callbacks = [
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

"""
Entrenar el modelo, en 5 épocas
Usar callbacks para extender el comportamiento durante el entrenamiento
Los callbacks se activan cuando se usan 20 épocas, por ejemplo
Para ver los datos en TensorBoard ejecutar: tensorboard --logdir=logs
Ir a: http://localhost:6006
"""
num_epochs = 20
model.fit(x_train, y_train, epochs = num_epochs,callbacks=callbacks,
          validation_data=(x_test, y_test))

#Validar el modelo
model.evaluate(x_test, y_test)

#Save weights to a TensorFlow Checkpoint file
#model.save_weights('./weights/my_model')

#Save weights to a HDF5 file
#model.save_weights('my_model.h5', save_format='h5')

#Save entire model to a HDF5 file
#model.save('my_entire_model.h5')

#Convertir modelo a formato TFLITE para usar en una app móvil Android o iOS
#converter = tf.lite.TFLiteConverter.from_keras_model_file("my_entire_model.h5")
#tflite_model = converter.convert()
#open("converted_model.tflite", "wb").write(tflite_model)

"""
Generar y guardar un diagrama del modelo en formato PNG
rankdir='TB' (top to bottom) gráfico vertical, "LR" (left to right) horizontal
El número que aparece al inicio del gráfico es incorrecto, debido a un bug de Keras
Debería indicar la cantidad de parámetros(weight, bias) del modelo
"""
#tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, rankdir='LR')

#Mostrar resumen del modelo
print(model.summary())

# Serialize a model to JSON format
json_string = model.to_json()
print (json_string)

#No disponible en TensorFlow 1.13
#tf.keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)