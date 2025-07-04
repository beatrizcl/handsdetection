import cv2
import mediapipe as mp

<<<<<<< HEAD
# Inicializa a webcam
webcam = cv2.VideoCapture(0)

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def detect_letter(landmarks):
    lm = {i: landmarks.landmark[i] for i in range(21)}

    def is_finger_extended(tip, pip):
        return lm[tip].y < lm[pip].y

    index_extended = is_finger_extended(8, 6)
    middle_extended = is_finger_extended(12, 10)
    ring_extended = is_finger_extended(16, 14)
    pinky_extended = is_finger_extended(20, 18)

    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    # Letra A: todos os dedos exceto polegar fechados
    if not index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "A"

    # Letra B: 4 dedos esticados, polegar dobrado para dentro perto do punho
    thumb_folded = abs(thumb_tip.x - wrist.x) < 0.05 and abs(thumb_tip.y - wrist.y) > 0.02

    if index_extended and middle_extended and ring_extended and pinky_extended and thumb_folded:
        return "B"

    # Letra C: polegar e indicador curvados formando arco
    def is_finger_curved(tip, pip):
        return lm[tip].y > lm[pip].y and (lm[tip].y - lm[pip].y) < 0.07

    thumb_curved = is_finger_curved(4, 3)
    index_curved = is_finger_curved(8, 6)

    thumb_index_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2) ** 0.5

    if thumb_curved and index_curved and thumb_index_dist > 0.08:
        return "C"

    return ""

# Loop 
while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    detected_letter = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            detected_letter = detect_letter(hand_landmarks)

    cv2.putText(frame, f'Letter: {detected_letter}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("ASL Translator - A, B, C", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

=======
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
>>>>>>> f99289ddbad122f4ffd57731863b5ba3da5a38b8
webcam.release()
cv2.destroyAllWindows()
