import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np

#Inicializa a captura de vídeo da primeira câmera disponível
cap = cv2.VideoCapture(0)

#Define um detector de mãos da MediaPipe que detecta no máximo uma mão por vez.
hands = mp.solutions.hands.Hands(max_num_hands=1)

#load_model('keras_model.h5'): Carrega um modelo de rede neural treinado previamente 
#(salvo como keras_model.h5).
classes = ['A', 'B', 'C', 'D', 'E']
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#Converte para RGB, redimensiona para 224x224 pixels, normaliza os valores
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 127.0 - 1
    return img

# Retorna a classe com a maior probabilidade de acordo com a predição.
def postprocess_output(prediction):
    index = np.argmax(prediction)
    return classes[index]

#Captura continuamente frames da câmera até que o usuário pressione 'q' para sair
while True:
    success, img = cap.read()
    if not success:
        break

#Converte o frame para o espaço de cores RGB, detecta landmarks 
#(pontos) das mãos usando o detector hands da MediaPipe.
    frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)
    handsPoints = results.multi_hand_landmarks
    h, w, _ = img.shape

    if handsPoints:
        for hand in handsPoints:
            x_max, y_max = 0, 0
            x_min, y_min = w, h
            for lm in hand.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_max, y_max = max(x_max, x), max(y_max, y)
                x_min, y_min = min(x_min, x), min(y_min, y)
            
            margin = 50
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)
            
            #Identifica a região da mão na imagem e extrai esta região.
            #Pré-processa a imagem da mão para prepará-la para a entrada no modelo de predição.
            imgCrop = img[y_min:y_max, x_min:x_max]
            imgPreprocessed = preprocess_image(imgCrop)
            data[0] = imgPreprocessed

            #Utiliza o modelo carregado para prever o gesto da mão na região de interesse.
            prediction = model.predict(data)
            gesture = postprocess_output(prediction)

            #Desenha um retângulo ao redor da mão detectada e exibe o gesto predito sobre a imagem
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(img, gesture, (x_min, y_min - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    #Mostra a imagem processada com as anotações visuais.
    cv2.imshow('Imagem', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#Este código combina detecção de mão usando MediaPipe, pré-processamento de imagem para a rede neural 
#e classificação usando um modelo treinado pelo Keras, 
#proporcionando assim um sistema básico de reconhecimento de gestos de mão em tempo real.
#https://teachablemachine.withgoogle.com/train

