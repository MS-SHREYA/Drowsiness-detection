# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python3 detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    # verticle eye landmarks (x,y coordinates)
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C) # eye aspect according to ear
    return ear
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', required = True, help = 'path to facial landmark predictor')
ap.add_argument('-a', '--alarm', type = str, default = '', help = 'path alarm.WAV file')
ap.add_argument('-w', '--webcam', type = int, default = 0, help = 'index of webcam on system')
args = vars(ap.parse_args())

eye_ar_thresh = 0.3
eye_ar_consec_frames = 48
counter = 0    # initializing counter
alarm_on = False

print('[INFO], loading facial landmark predictor')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args['shape_predictor'])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

print('[info] starting video screen thread')
video_stream = VideoStream(src = args['webcam']).start()
time.sleep(1.0)

while True : 
    frame = video_stream.read()
    frame = imutils.resize(frame,width = 450)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rect = detector(gray,0)

    for i in rect:
        shape = predictor(gray, i)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0  # average eye aspect ratio of both eyes
        
        leftEyeHull = cv2.convexHull(left_eye)
        rightEyeHull = cv2.convexHull(right_eye)

        cv2.drawContours(frame,[leftEyeHull], -1, (0,0,255), 1)
        cv2.drawContours(frame,[rightEyeHull], -1, (0,0,255), 1)

        if ear < eye_ar_thresh:
            counter += 1
        
            if counter >= eye_ar_consec_frames:
                if not alarm_on:
                    alarm_on = True
                    if args['alarm'] != '':
                        T = Thread(target = sound_alarm, args = (args['alarm'],))
                        T.daemon = True
                        T.start()
                cv2.putText(frame, 'drowsiness alert', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2) 
            else:
                counter = 0
                alarm_on = False
            cv2.putText(frame,'ear:{:.2f}'.format(ear), (300,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
cv2.destroyAllWindows()
video_stream.stop()
                