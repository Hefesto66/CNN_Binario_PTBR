from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# carrega a rede treinada
model = load_model('/home/carlin/Imagens/CNN.h5')

# carrega as imagens de teste
img1 = Image.open('/home/carlin/Imagens/testeCNN/image1.jpg')
img2 = Image.open('/home/carlin/Imagens/testeCNN/image2.jpg')
img3 = Image.open('/home/carlin/Imagens/testeCNN/image3.jpg')
img4 = Image.open('/home/carlin/Imagens/testeCNN/image4.jpg')
img5 = Image.open('/home/carlin/Imagens/testeCNN/image5.jpg')
img6 = Image.open('/home/carlin/Imagens/testeCNN/image6.jpg')

# converte as imagens para escala de cinza e redimensiona para o tamanho esperado pela rede neural
img1 = img1.convert('L').resize((177, 144))
img2 = img2.convert('L').resize((177, 144))
img3 = img3.convert('L').resize((177, 144))
img4 = img4.convert('L').resize((177, 144))
img5 = img5.convert('L').resize((177, 144))
img6 = img6.convert('L').resize((177, 144))

# converte as imagens de teste em arrays numpy
x1 = image.img_to_array(img1)
x2 = image.img_to_array(img2)
x3 = image.img_to_array(img3)
x4 = image.img_to_array(img4)
x5 = image.img_to_array(img5)
x6 = image.img_to_array(img6)

# expande as dimensões dos arrays para que eles tenham forma (1, 144, 177, 1)
x1 = np.expand_dims(x1, axis=0)
x2 = np.expand_dims(x2, axis=0)
x3 = np.expand_dims(x3, axis=0)
x4 = np.expand_dims(x4, axis=0)
x5 = np.expand_dims(x5, axis=0)
x6 = np.expand_dims(x6, axis=0)

# normaliza os arrays
x1 = x1.astype('float32') / 255.0
x2 = x2.astype('float32') / 255.0
x3 = x3.astype('float32') / 255.0
x4 = x4.astype('float32') / 255.0
x5 = x5.astype('float32') / 255.0
x6 = x6.astype('float32') / 255.0

# faz a predição das classes das imagens usando a rede treinada
preds = model.predict(np.concatenate((x1, x2, x3, x4, x5, x6)))

# imprime uma mensagem indicando se cada imagem é de uma vaca ou não
for i, pred in enumerate(preds):
    if pred > 0.5:
        print(f"A imagem {i+1} é de uma vaca")
    else:
        print(f"A imagem {i+1} não é de uma vaca")