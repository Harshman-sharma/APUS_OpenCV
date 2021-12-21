# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:45:13 2018

@author: Harshman
"""

import numpy as np
import cv2

stop_cascade = cv2.CascadeClassifier('AutoRCCar-master\computer\cascade_xml\stop_sign.xml')

cap = cv2.VideoCapture(0)

while(True):
    
    
    ret,frame = cap.read()
    img = cv2.GaussianBlur(frame,(25,25),0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_range = np.array([150,50,50], dtype = np.uint8)
    upper_range = np.array([200,255,255], dtype = np.uint8)
    img= cv2.inRange(hsv, lower_range, upper_range)
    can_edge = cv2.Canny(img,50,150)#low_threshold=50,upper_threshold=150
	
    circles = cv2.HoughCircles(can_edge, cv2.HOUGH_GRADIENT, 1, 10, param1=10, param2=35, minRadius=1, maxRadius=200)
    
    if circles is not None:
        
        circles = np.uint16(np.around(circles))
        
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(frame, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(frame, center, radius, (255, 0, 255), 3)
            
        print("Stop! There's a red light!")
    
    cv2.imshow("image",can_edge)
	
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sign = stop_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    
    
    for (x, y, w, h) in sign:
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        img_item = "stop.png"
        cv2.imwrite(img_item, roi_color)
        
        color = (255,0,0)
        stroke =2
        cv2.rectangle(frame,(x,y), (x+w,y+h), color, stroke)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = 'stop'
        color = (255,255,0)
        stroke = 2
        cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        print("stop")
        
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(20) & 0xFF== ord('q'):
        break
    # k = cv2.waitKey(20)
    # if k == c'q':
        # break

cap.release()
cv2.destroyAllWindows()