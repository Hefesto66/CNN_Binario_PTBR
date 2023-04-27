
# 2023, Carlos Silva
# Contato: carlos.silva@axautomacao.com.br

# Importar bibliotecas necessárias
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Verificar se a GPU está disponível
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('GPU disponível:', physical_devices[0])
else:
    print('Nenhuma GPU disponível.')

tf.test.gpu_device_name()

# Definir parâmetros
batch_size = 32
epochs = 20
learning_rate = 0.001

# Carregar dados
diretorio_dados = '/home/carlin/Imagens/dados'
try:
    dados_vacas = np.load(os.path.join(diretorio_dados, 'vaca.npz'))['dados']
    dados_sem_vacas = np.load(os.path.join(diretorio_dados, 'semvaca.npz'))['dados']
except:
    print("Erro ao carregar os dados.")

# Adicionar labels aos dados 
x_vacas = dados_vacas
y_vacas = np.ones(len(x_vacas), dtype=np.int32) # 1 representa vacas
x_sem_vacas = dados_sem_vacas
y_sem_vacas = np.zeros(len(x_sem_vacas), dtype=np.int32) # 0 representa falta de vacas

# Concatenar dados
x = np.concatenate((x_vacas, x_sem_vacas), axis=0)
y = np.concatenate((y_vacas, y_sem_vacas), axis=0)

# Dividir dados em conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalizar dados
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Adicionar dimensão para número de canais
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Definir modelo
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(144, 177, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compilar modelo
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              metrics=['accuracy'])

# Criar geradores de imagem para aumentar o conjunto de dados de treinamento
    #Aplicando rotação, zoom, deslocamento horizontal e vertical e flip horizontal.
train_datagen = ImageDataGenerator(rotation_range=20, 
                                   zoom_range=0.2, 
                                   width_shift_range=0.2, 
                                   height_shift_range=0.2, 
                                   horizontal_flip=True)
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)

# Treinar modelo
history = model.fit(train_generator,
                    steps_per_epoch=len(x_train) // batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))

# Avaliar modelo
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Salvar modelo
model_dir = '/home/carlin/Imagens'
model_filename = 'modelo.h5'
model.save(os.path.join(model_dir, model_filename))

# Plotar acurácia e perda ao longo das épocas
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy/Loss')
plt.legend(loc='lower right')
plt.show()

# Mensagem de confirmação
print("O treinamento e avaliação do modelo foram concluídos com sucesso!")