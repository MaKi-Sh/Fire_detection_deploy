import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)

# Beep 3 times
for i in range(3):
    GPIO.output(7, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(7, GPIO.LOW)
    time.sleep(0.5)

GPIO.cleanup()
