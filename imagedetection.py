import cv2
import mediapipe as mp

# conect the webcam to python
webcam = cv2.VideoCapture(0) # starts a conection with the webcam

# inicializing the mediapipe
hands_recogner = mp.solutions.hands
draw_mp = mp.solutions.drawing_utils
hands = hands_recogner.Hands()

# what happen if the webcam inst open
if webcam.isOpened():
    # read the webcam (webcam.read())
    validate, frame = webcam.read()
    # Undestand webcam.read() -> get the frames
    # loop 
    while validate:
        # get the next frame
        validate, frame = webcam.read()
        
        # convert BGR (opencv pattern) to RGB (mediapipe pattern)
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Draw the hands
        hands_list = hands.process(frameRGB)
        if hands_list.multi_hand_landmarks:
            for mao in hands_list.multi_hand_landmarks:
                draw_mp.draw_landmarks(frame, hands, hands_recogner.HAND_CONNECTIONS)

        # show the frame of webcam that python is looking
        cv2.imshow("Video", frame)
        # ask to python wait -> in a smart way xD 
        key = cv2.waitKey(2)
        # ask to stop when you press the key 27 is ESC 
        if key == 27:
            break

#  Stop
webcam.release()
cv2.destroyAllWindows()
