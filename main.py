#https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
import time

import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture("http://192.168.10.152:4747/video")
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1,
    static_image_mode=False) as hands:
  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    inicio=time.time()
    results = hands.process(image)
    #print(f'fps {1/(time.time()-inicio)}')
    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      image_height, image_width, _ = image.shape
      for hand_landmarks in results.multi_hand_landmarks:
        dedao_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
        dedao_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
        dedao = (int(dedao_x), int(dedao_y))
        indicador_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
        indicador_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
        indicador = (int(indicador_x), int(indicador_y))
        print(indicador)
        cv2.circle(image,indicador,2,(255,255,0),cv2.FILLED)
        cv2.circle(image, dedao, 2, (255, 255, 0), cv2.FILLED)

        distancia = int(np.sqrt((indicador_x - dedao_x) ** 2 + (indicador_y - dedao_y) ** 2))
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(30) & 0xFF == 27:
      break
cap.release()