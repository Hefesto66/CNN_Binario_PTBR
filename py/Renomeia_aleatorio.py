import os
import random
import string

# Definir o caminho para a pasta com as imagens
caminho_pasta = "/home/carlin/Imagens/semvaca/teste177x144/sem vaca"

# Listar todos os arquivos na pasta
arquivos = os.listdir(caminho_pasta)

# Loop para renomear cada arquivo
for arquivo in arquivos:
    # Verificar se o arquivo é uma imagem
    if arquivo.endswith(".jpg") or arquivo.endswith(".jpeg") or arquivo.endswith(".png"):
        # Gerar um novo nome aleatório para o arquivo
        novo_nome = "vaca_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10)) + ".jpg"

        # Renomear o arquivo com o novo nome
        os.rename(os.path.join(caminho_pasta, arquivo), os.path.join(caminho_pasta, novo_nome))
