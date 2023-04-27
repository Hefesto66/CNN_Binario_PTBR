import os
from shutil import copyfile

# Especifica o diretório onde as imagens estão
diretorio_origem = '/home/carlin/Imagens/semvaca/cavalosT'

# Especifica o diretório onde as cópias das imagens serão criadas
diretorio_destino = '/home/carlin/Imagens/semvaca/copias'

# Loop através de todas as imagens numeradas de 1 a 13
for i in range(1, 14):
    nome_arquivo_origem = str(i) + '.jpg'
    caminho_arquivo_origem = os.path.join(diretorio_origem, nome_arquivo_origem)
    
    # Verifica se o arquivo de origem existe antes de copiá-lo
    if os.path.exists(caminho_arquivo_origem):
        
        # Loop para criar 3 cópias de cada imagem
        for j in range(1, 4):
            nome_arquivo_destino = str(i) + '_' + str(j) + '.jpg'
            caminho_arquivo_destino = os.path.join(diretorio_destino, nome_arquivo_destino)
            copyfile(caminho_arquivo_origem, caminho_arquivo_destino)
            
        # Loop interno para criar 2 cópias adicionais de cada imagem
        for k in range(4, 6):
            nome_arquivo_destino = str(i) + '_' + str(k) + '.jpg'
            caminho_arquivo_destino = os.path.join(diretorio_destino, nome_arquivo_destino)
            copyfile(caminho_arquivo_origem, caminho_arquivo_destino)
    else:
        print(f"Arquivo {caminho_arquivo_origem} não encontrado.")