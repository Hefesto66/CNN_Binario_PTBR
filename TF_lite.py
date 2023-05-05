import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

# Carrega o modelo a partir do arquivo .h5
model = load_model('/home/carlin/Documentos/CNN_final.h5')

# Define o otimizador e a função de perda para o modelo
optimizer = keras.optimizers.Adam()
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

# Compila o modelo com o otimizador e a função de perda
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# Define o conversor de precisão para 8 bits
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Define o conjunto de calibração para a conversão
num_calibration_steps = 100
input_shape = (1, 144, 176, 1)

def representative_dataset():
    for _ in range(num_calibration_steps):
        # Carrega um lote de dados de entrada
        x = tf.random.normal(input_shape)
        yield [x]

converter.representative_dataset = representative_dataset

# Converte o modelo para o formato TFLite com precisão de 8 bits
tflite_model = converter.convert()

# Salva o modelo convertido em um arquivo .tflite
with open('/home/carlin/Documentos/CNN_final.tflite', 'wb') as f:
    f.write(tflite_model)