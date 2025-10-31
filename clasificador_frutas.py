import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Datos
RUTA_TRAIN = 'archive'
LOTE = 32

# Data Aumentation
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalizar 0-255 → 0-1
    rotation_range=20,        # Rotar 20°
    width_shift_range=0.2,    # Mover horizontal
    height_shift_range=0.2,   # Mover vertical
    zoom_range=0.2,           # Zoom in/out
    horizontal_flip=True,     # Espejo horizontal
    validation_split=0.2      # 80/20 auto-split
)

# Generadores (carga + datos aumentados + lotes)
train_gen = train_datagen.flow_from_directory(
    RUTA_TRAIN,
    target_size=(320, 258),
    batch_size=LOTE,
    class_mode='categorical', # softmax
    subset='training' # 80%
)

val_gen = train_datagen.flow_from_directory(
    RUTA_TRAIN,
    target_size=(320, 258),
    batch_size=LOTE,
    class_mode='categorical',
    subset='validation' # 80%
)

print("Clases detectadas:", list(train_gen.class_indices.keys()))

# Crear el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(320, 258, 3), activation='relu'), # convolucion
    tf.keras.layers.MaxPooling2D(2,2), # agrupacion

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # convolucion
    tf.keras.layers.MaxPooling2D(2,2), # agrupacion

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'), # convolucion
    tf.keras.layers.MaxPooling2D(2,2), # agrupacion

    tf.keras.layers.Dropout(0.5), # 50% de desactivar neuronas
    tf.keras.layers.Flatten(), # aplanar
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(15, activation='softmax')
])

# Compilar el modelo
modelo.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
modelo.summary()

# Entrenar el modelos
historial = modelo.fit(
    train_gen,
    epochs=30,
    validation_data=val_gen,
    steps_per_epoch=len(train_gen)
)

# Realizar prueba
print("\n=== EVALUACIÓN DEL MODELO ===")

# Evaluación en conjunto de validación
loss, accuracy = modelo.evaluate(val_gen, verbose=0)
print(f"Pérdida en validación: {loss:.4f}")
print(f"Precisión en validación: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Predicciones en un lote de validación
print("\n=== PREDICCIONES DE MUESTRA ===")
batch_images, batch_labels = next(val_gen)
predictions = modelo.predict(batch_images)

# Obtener nombres de clases
class_names = list(train_gen.class_indices.keys())

# Mostrar predicciones de las primeras 5 imágenes
for i in range(min(5, len(batch_images))):
    predicted_class = np.argmax(predictions[i])
    actual_class = np.argmax(batch_labels[i])
    confidence = predictions[i][predicted_class]
    
    print(f"Imagen {i+1}:")
    print(f"  Predicción: {class_names[predicted_class]} (confianza: {confidence:.4f})")
    print(f"  Real: {class_names[actual_class]}")
    print(f"  ¿Correcto?: {'✓' if predicted_class == actual_class else '✗'}")
    print()

# Exportar el modelo entrenado
modelo.save('frutas_cnn.h5')