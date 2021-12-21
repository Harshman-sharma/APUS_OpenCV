from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import numpy as np
import math
import serial

arduino = serial.Serial('/dev/ttyUSB0', 9600)

camera = PiCamera()
camera.resolution = (640,480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640,480))

time.sleep(0.1)
stop_cascade = cv2.CascadeClassifier('/home/pi/Documents/AutoRCCar-master/computer/cascade_xml/stop_sign.xml')
stop2_cascade = cv2.CascadeClassifier('/home/pi/Documents/AutoRCCar-master/computer/cascade_xml/my_traffic_light.xml')

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):



    #PART1

    image = frame.array
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sign = stop_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(150, 150))
    for (x, y, w, h) in sign:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        img_item = "stop.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (255,0,0)
        stroke =2
        cv2.rectangle(image,(x,y), (x+w,y+h), color, stroke)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = 'stop'
        color = (255,255,0)
        stroke = 2
        cv2.putText(image, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        print("stop")
        arduino.write('stop')
        pointer=1
        for i in range (10):
            time.sleep(0.5)
            print('stop')
            arduino.write('stop')
        for i in range (4):
            time.sleep(0.5)
            print('straight')
            arduino.write('straight')
        pointer=0
        
    pointer=0



    #PART2 (RAW)
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ##blurred = cv2.GaussianBlur(image,(25,25),0)
	blurred = cv2.GaussianBlur(gray,(25,25),0)
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #lower_range = np.array([150,100,100], dtype = np.uint8)#[150,100,100]
    #upper_range = np.array([200,255,255], dtype = np.uint8)#[200,255,255]
    #img= cv2.inRange(hsv, lower_range, upper_range)
    #edged = cv2.Canny(img,50,150)
    edged2 = cv2.Canny(blurred,50,150)
    #
    #circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 10, param1=10, param2=35, minRadius=1, maxRadius=200)
    #
    #if circles is not None:
     #   circles = np.uint16(np.around(circles))
      #  for i in circles[0, :]:
       #     center = (i[0], i[1])
        #    # circle center
         #   cv2.circle(image, center, 1, (0, 100, 100), 3)
          #  # circle outline
           # radius = i[2]
            #cv2.circle(image, center, radius, (255, 0, 255), 3)
        #print('stop')
        #arduino.write('stop')
    
    #cv2.imshow("image",edged)



    #PART2
    light = stop2_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5, minSize=(150, 150))
    for (x, y, w, h) in light:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        
        img_item = "stop.png"
        cv2.imwrite(img_item, roi_gray)
        
        color = (255,0,0)
        stroke =2
        cv2.rectangle(image,(x,y), (x+w,y+h), color, stroke)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = 'stop'
        color = (255,255,0)
        stroke = 2
        cv2.putText(image, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        print("stop")
        arduino.write('stop')
           
    #PART3

    theta=0
    minLineLength = 5
    maxLineGap = 10
    lines = cv2.HoughLinesP(edged2,1,np.pi/180,10,minLineLength,maxLineGap)

    if(lines !=None):
        for x in range(0, len(lines)):
           for x1,y1,x2,y2 in lines[x]:
               #cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
               theta=theta+math.atan2((y2-y1),(x2-x1))

    #print(theta)GPIO pins were connected to arduino for servo steering control
    threshold=6
    if(theta>threshold)and pointer!=1:
        print("left")
        arduino.write('left')

        if(theta>threshold)and pointer==1:
            print("stop")
            time.sleep(5)
            print("left")
            arduino.write('left')
            
            
    if(theta<-threshold)and pointer!=1:
        print("right")
        arduino.write('right')

        if(theta<-threshold)and pointer==1:
            print("stop")
            time.sleep(5)
            print("right")
            arduino.write('right')
            
        
    if(abs(theta)<threshold)and pointer!=1:
        print("straight")
        arduino.write('straight')
        
        if(abs(theta)<threshold)and pointer==1:
            print("stop")
            time.sleep(5)
            print("straight")
            arduino.write('straight')
            
            
    #else:
     #   print('stop')
      #  arduino.write('stop')
    theta=0

    cv2.imshow('frame', image)
    rawCapture.truncate(0)
    if cv2.waitKey(20) & 0xFF== ord('q'):
        break

cv2.destroyAllWindows()
arduino.close
