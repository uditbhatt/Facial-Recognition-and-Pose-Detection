# -*- coding: utf-8 -*-
"""
program for creating a dataset for program
Created on Thu Apr 18 22:20:05 2019
@author: manishluthyagi
"""

import cv2 as cv
import os as dire

casc_classifier = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
capture_frame = cv.VideoCapture(0)
count_no = 0
person_name = input("\n Person name in Video stream :\t >")

dire.mkdir("train/"+person_name)
dire.mkdir("test/"+person_name)
while True:
    
    B_value ,frame_img = capture_frame.read()
    gray_img = cv.cvtColor(frame_img, cv.COLOR_BGR2GRAY)

    face_set = casc_classifier.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 5, minSize = (30, 30), flags = cv.CASCADE_SCALE_IMAGE )
    
    for (x1, y1, x2, y2) in face_set:
        cv.rectangle(frame_img, (x1, y1), (x1+x2, y1+y2), (0, 255, 255), 1)
        
    if count_no < 250:
        cv.imwrite("train/"+person_name+'/'+str(count_no)+ ".jpg", frame_img)
    else:
       cv.imwrite("test/"+person_name+'/'+str(count_no)+ ".jpg", frame_img)
        
    cv.imshow(person_name,frame_img)
    count_no = count_no + 1
    if count_no > 350:
        break
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
capture_frame.release()
cv.destroyAllWindows()
