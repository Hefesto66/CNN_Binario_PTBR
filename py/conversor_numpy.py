import numpy as np
import os
from PIL import Image

diretorio_imagens = '/home/carlin/Imagens/'
diretorio_dados = '/home/carlin/Imagens/dataset'
classes = ['vaca']

tamanho_imagem = (177, 144)

dados = []
rotulos = []

for classe in classes:
    diretorio_classe = os.path.join(diretorio_imagens, classe)
    for imagem in os.listdir(diretorio_classe):
        imagem_path = os.path.join(diretorio_classe, imagem)
        imagem = Image.open(imagem_path)
        matriz = np.array(imagem)
        dados.append(matriz)
        rotulos.append(classes.index(classe))
        
dados = np.array(dados)
rotulos = np.array(rotulos)

if not os.path.exists(diretorio_dados):
    os.makedirs(diretorio_dados)

np.savez(os.path.join(diretorio_dados, 'vaca.npz'), dados=dados, rotulos=rotulos)