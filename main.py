import cv2
import mediapipe as mp
from time import *
import random
import numpy as np


class HandDetectorClass:
    def __init__(self, mode=False, maxHands=2, detectConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectConf = detectConf
        self.trackConf = trackConf

        self.hands = mp.solutions.hands.Hands(self.mode, self.maxHands, self.detectConf, self.trackConf)
        self.draw = mp.solutions.drawing_utils

        self.ct = 0
        self.pred = ''

    def findHands(self, screen, draw=True):
        rgb = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)

        if self.results.multi_hand_landmarks:
            for lms in self.results.multi_hand_landmarks:
                if draw:
                    self.draw.draw_landmarks(screen, lms, mp.solutions.hands.HAND_CONNECTIONS)

    def showHands(self, screen):
        if self.results.multi_hand_landmarks:
            for lms in self.results.multi_hand_landmarks:
                for id, lm in enumerate(lms.landmark):
                    h, w, _ = screen.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    mask = np.zeros(screen.shape[:2], dtype="uint8")
                    cv2.circle(mask, (cx, cy), 10, 255, -1)
                    masked = cv2.bitwise_and(screen, screen, mask=mask)

    def posHands(self, screen, draw=True, no=0):
        lml = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[no]
            for id, lm in enumerate(hand.landmark):
                h, w, _ = screen.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                lml.append([id, cx, cy])

                if id == 8 and draw:
                    cv2.circle(screen, (cx, cy), 15, (225, 20, 20), cv2.FILLED)

        return lml

    def detectOpen(self, lml):
        handOpen = []
        tips = [8, 12, 16, 20]
        if len(lml) > 0:
            if lml[4][1] < lml[2][1]:
                handOpen.append(0)
            else:
                handOpen.append(1)

            for i in tips:
                if lml[i][2] < lml[i - 2][2]:
                    handOpen.append(1)
                else:
                    handOpen.append(0)

        return handOpen

    def detectSPS(self, screen, handOpen, last):
        pt = ''
        
        if self.results.multi_hand_landmarks:
            if not 0 in handOpen:
                pt = 'Paper'
            elif handOpen.count(1) <= 1:
                pt = 'Stone'
            elif handOpen[1] == 1 and handOpen[2] == 1 and handOpen[3] == 0 and handOpen[4] == 0:
                pt = 'Scissor'

            if last == pt or last == '':
                self.ct += 1
            else:
                self.ct = 0

            if self.pred == pt or self.ct == 15:
                cv2.putText(screen, pt, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (225, 50, 50), 2)
                self.ct = 0
                self.pred = pt

        else:
            self.pred = ''

        return pt


def bot(screen, hum):
    _, w, _ = screen.shape
    botSPS = random.choice(['Paper', 'Stone', 'Scissor'])
    w -= 100 

    if botSPS == 'Scissor':
        w -= 20

    if hum:
        cv2.putText(screen, botSPS, (w, 30), cv2.FONT_HERSHEY_PLAIN, 2, (225, 50, 50), 2)
        return False, botSPS

    return True, ''

def win(bp, p, screen):
    winText = ''
    if bp == 'Paper' and p == 'Stone':
        winText = 'Bot wins!'
    elif bp == 'Stone' and p == 'Scissor':
        winText = 'Bot wins!'
    elif bp == 'Scissor' and p == 'Paper':
        winText = 'Bot wins!'
    elif bp == 'Stone' and p == 'Paper':
        winText = 'User wins!'
    elif bp == 'Scissor' and p == 'Stone':
        winText = 'User wins!'
    elif bp == 'Paper' and p == 'Scissor':
        winText = 'User wins!'
    else:
        winText = 'Draw'

    h, w, _ = screen.shape
    cv2.putText(screen, winText, (w // 2 - 100, h // 2), cv2.FONT_HERSHEY_PLAIN, 3, (225, 50, 50), 3)


def main():
    cap = cv2.VideoCapture(0)

    detector = HandDetectorClass()
    user = ''

    run = True    
    while run:
        _, screen = cap.read()

        detector.findHands(screen, False)
        lml = detector.posHands(screen, draw=False)
        ho = detector.detectOpen(lml)
        user = detector.detectSPS(screen, ho, user)

        run, botPred = bot(screen, detector.pred)
        if not run:
            win(botPred, detector.pred, screen)
        
        cv2.imshow('Camera Screen', screen)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sleep(5)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
