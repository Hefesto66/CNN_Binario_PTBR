from PIL import Image
import os

# Define o diretório onde estão as imagens
diretorio = '/home/carlin/Imagens/semvaca/cavalos'

# Define o diretório onde as novas imagens serão salvas
diretorio_destino = '/home/carlin/Imagens/semvaca/cavalosT'

# Cria o diretório se ele não existir
if not os.path.exists(diretorio_destino):
    os.makedirs(diretorio_destino)

# Percorre todas as imagens do diretório
for filename in os.listdir(diretorio):
    # Verifica se o arquivo é uma imagem
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Abre a imagem
        img = Image.open(os.path.join(diretorio, filename))
        # Converte para escala de cinza
        img = img.convert('L')
        # Redimensiona para resolução desejada
        img = img.resize((177, 144))
        # Salva a imagem no diretório de destino
        img.save(os.path.join(diretorio_destino, 'p_' + filename))