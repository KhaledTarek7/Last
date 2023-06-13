import RPi.GPIO as GPIO
import time
import cv2
import imutils
import numpy as np
import pytesseract

from picamera.array import PiRGBArray
from picamera import PiCamera

PIR_sensor2 = 23
PIR_sensor1 = 24
m12 = 5
m11 = 6
pwm = 16
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(m11, GPIO.OUT)
GPIO.setup(m12, GPIO.OUT)
GPIO.output(m11, GPIO.LOW)
GPIO.output(m12, GPIO.LOW)
GPIO.setup(PIR_sensor1, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(PIR_sensor2, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(pwm, GPIO.OUT)

pwm = GPIO.PWM(16, 90)
pwm.start(80)


def Khaled():
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)
        
        if key == ord("s"):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(gray, 30, 200)
            
            cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
            screenCnt = None
            
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                
                if len(approx) == 4:
                    screenCnt = approx
                    break
            
            if screenCnt is None:
                detected = 0
            else:
                detected = 1
            
            if detected == 1:
                cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
                
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1,)
            new_image = cv2.bitwise_and(image, image, mask=mask)
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))
            Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
            
            text = pytesseract.image_to_string(Cropped, config='--psm 11')
            cleanText = "".join(char.upper() for char in text if char.isalnum() or char.isspace())
            
            print("Detected Number is:", cleanText)
            cv2.imshow('image', image)
            cv2.imshow('Cropped', Cropped)
            cv2.waitKey(1)
            break
    
    cv2.destroyAllWindows()
    return cleanText


gate_open = False  # Initially, the gate is closed.

while True:
    time.sleep(2)
    
    # Check sensor 1
    if GPIO.input(PIR_sensor1):
        print("Welcome To Khaleds Home")
        car = Khaled()
        print(car)
        
        if car.strip() == "KAU 3881":
            time.sleep(2)
            GPIO.output(m11, GPIO.HIGH)  # gate opening
            GPIO.output(m12, GPIO.LOW)
            gate_open = True  # Gate is now open
            time.sleep(2)  # delay for simulation
            
            GPIO.output(m11, GPIO.LOW)  # gate stop for a while
            GPIO.output(m12, GPIO.LOW)
            time.sleep(5)
    
    # Check sensor 2
    if GPIO.input(PIR_sensor2):
        time.sleep(2)
        
        if gate_open:
            # This means a car that entered is now leaving, so close the gate
            print("Closing gate")
            GPIO.output(m11, GPIO.LOW)
            GPIO.output(m12, GPIO.HIGH)
            gate_open = False  # Gate is now closed
        else:
            # This means a car from inside wants to go out, so open the gate
            print("Opening gate for exit")
            GPIO.output(m11, GPIO.HIGH)
            GPIO.output(m12, GPIO.LOW)
            gate_open = True  # Gate is now open
        
        time.sleep(2)
        GPIO.output(m11, GPIO.LOW)
        GPIO.output(m12, GPIO.LOW)

    time.sleep(5)
