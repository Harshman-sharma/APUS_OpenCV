# APUS_OpenCV
A python program to demonstrate how OpenCV and raspberry Pi 3 Model B can be used to built a basic Self Driving car model.

The code in APUS.py is divided into 3 parts, for 3 different purposes, ie, 
1) Object detection : Stop sign detection, 
2) Object detection : Traffic lights detection, 
3) Hough lines : To Detect the lane in which the car is moving and steer accordingly.

OUTPUTS:

![image](https://github.com/Harshman-sharma/APUS_OpenCV/assets/44753624/cbbddeba-498a-405e-a75b-4791d960b2eb)

![image](https://github.com/Harshman-sharma/APUS_OpenCV/assets/44753624/a4606079-73bd-4826-98e6-1eb8ee0a5cbf)


Components needed : 
1) 1 Raspberry Pi (any model)
2) 1 PiCam module
3) 1 Arduino (any model)
4) 1 Servo
5) 2 Motors
6) 1 car chassis
7) 1 power source
8) Jumper wires

NOTE : In this case, the frontal tyres of car model are not connected to any motors and so the steering module is created by connecting them to a servo which turns a few degrees to steer both wheels in the desired direction.

Code might differ a bit for creators who have incorporated variation in steering mechanism.

#NOTE: Although not compulsary, but the code also gives way to establish user access to a serial connection which could be used to trasfer data between an Arduino device and our Raspberry Pi. This serial connection could be used to control/analyse different components via Arduino and then share that data with the Raspberry Pi.
