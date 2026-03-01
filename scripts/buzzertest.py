import Jetson.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(33, GPIO.OUT)

# Create PWM at 1kHz frequency for passive buzzer
pwm = GPIO.PWM(33, 1000)
pwm.start(50)  # 50% duty cycle

# Beep 3 times
for i in range(3):
    time.sleep(0.3)  # Sound on
    pwm.stop()
    time.sleep(0.3)  # Sound off

pwm.stop()
GPIO.cleanup()
