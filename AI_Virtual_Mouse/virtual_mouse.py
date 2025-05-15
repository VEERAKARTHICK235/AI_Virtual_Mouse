import cv2
import numpy as np
import mediapipe as mp
import autopy
import math

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

screen_width, screen_height = autopy.screen.size()
prev_x, prev_y = 0, 0
smoothening = 7
left_click_down = False
right_click_down = False
double_click_triggered = False

def get_landmark_position(hand_landmarks, index):
    lm = hand_landmarks.landmark[index]
    return int(lm.x * wCam), int(lm.y * hCam)

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []
    for tip in tips:
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[tip - 2].y
        fingers.append(tip_y < pip_y)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        handLms = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        index_tip = get_landmark_position(handLms, 8)
        middle_tip = get_landmark_position(handLms, 12)
        thumb_tip = get_landmark_position(handLms, 4)

        # Cursor movement with index finger
        x3 = np.interp(index_tip[0], (100, wCam - 100), (0, screen_width))
        y3 = np.interp(index_tip[1], (100, hCam - 100), (0, screen_height))
        curr_x = prev_x + (x3 - prev_x) / smoothening
        curr_y = prev_y + (y3 - prev_y) / smoothening
        autopy.mouse.move(screen_width - curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

        # Left Click (Thumb + Index finger pinch)
        dist_thumb_index = calculate_distance(index_tip, thumb_tip)
        if dist_thumb_index < 40:
            if not left_click_down:
                left_click_down = True
                autopy.mouse.click()
                cv2.putText(img, "Left Click", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            left_click_down = False

        # Right Click (Index + Middle finger pinch)
        dist_index_middle = calculate_distance(index_tip, middle_tip)
        if dist_index_middle < 40:
            if not right_click_down:
                right_click_down = True
                autopy.mouse.click(button=autopy.mouse.Button.RIGHT)
                cv2.putText(img, "Right Click", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        else:
            right_click_down = False

        # Double Click (All 5 fingers open)
        finger_states = fingers_up(handLms)
        if all(finger_states):
            if not double_click_triggered:
                double_click_triggered = True
                autopy.mouse.click()
                autopy.mouse.click()
                cv2.putText(img, "Double Click", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        else:
            double_click_triggered = False

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
