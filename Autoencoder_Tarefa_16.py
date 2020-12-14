"""
Redes Neurais - Autoencoder
Aprendizagem não supervisionada
Redução de dimensionalidade - Tarefa
Base de dados CIFAR10
Construção de um autoencoder para redimensionamento, codificação e decodificação de imagens

"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Model, Sequential
from keras.layers import Input, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


(previsores_treinamento, _), (previsores_teste, _) = cifar10.load_data()

previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:])))
previsores_teste = previsores_teste.reshape((len(previsores_teste), np.prod(previsores_teste.shape[1:])))


""" Algoritmo para o treinamento """
# 3072 - 1536 - 768 - 1536 - 3072
autoencoder = Sequential()

# Encode
autoencoder.add(Dense(units=1536, activation='relu', input_dim=3072))
autoencoder.add(Dense(units=768, activation='relu'))


# Decode
autoencoder.add(Dense(units=1536, activation='relu'))
autoencoder.add(Dense(units=3072, activation='sigmoid'))
print(autoencoder.summary())


autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, epochs=100, batch_size=256,
                validation_data=(previsores_teste, previsores_teste))


""" Capturar o codificador """
# Criação manual da rede, capturando as camadas ocultas (que são as imagens codificadas)
dimensao_original = Input(shape=(3072,))
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]
encoder = Model(dimensao_original, camada_encoder2(camada_encoder1(dimensao_original)))
print(encoder.summary())

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)


""" Visualização das imagens """
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)
plt.figure(figsize=(18, 18))
for i, indice_imagem in enumerate(imagens_teste):

    # Imagem original
    eixo = plt.subplot(10, 10, i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(32, 32, 3))
    plt.xticks(())
    plt.yticks(())

    # Imagem codificada
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(16, 16, 3))
    plt.xticks(())
    plt.yticks(())

    # Imagem reconstruída
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(32, 32, 3))
    plt.xticks(())
    plt.yticks(())

plt.show()



print('\nFim')