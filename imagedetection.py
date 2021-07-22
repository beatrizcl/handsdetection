import cv2
import mediapipe as mp

# opencv - integrar webcam e usar python para trabalhar imagem
# mediapipe - reconhecer as mãos usando solutions hands



webcam = cv2.VideoCapture(0) # cria uma conexão com a web cam, o número 0 significa que estou usando a camera zero (câmera nativa do notebook)

reconhecimento_maos = mp.solutions.hands #inicializando mediapipe - solution hands: Reconhecimento de mãos
desenho_mp = mp.solutions.drawing_utils
maos = reconhecimento_maos.hands

if webcam.isOpened(): # o que acontece se a camera não abrir 
    validacao, frame = webcam.read()
    while validacao: #loop infinito
        validacao, frame = webcam.read()

        frameRGB = cv2.cvtColor(frame, cv2.COLOR.BRG2RGB) # converte formato bgr para rgb
        # desenhar a mão
        lista_maos = maos.process(frameRGB)
        if lista_maos.multi_hand_landmarks:
            for mao in lista_maos.multi_hand_landmarks:
                print(mao.landmark)
                desenho_mp.draw_landmarks(frame, mao, reconhecimento_maos.HAND_CONNECTIONS)


        cv2.imshow("Video", frame)
        tecla = cv2.waitKey(2) # 60 segundos -> 2 milisegundos = 30fps
        
        if tecla == 27: # fazer o código parar -- esc é o 27 na tabela scsi
            break
            
        
webcam.release() # Fecha a webcam