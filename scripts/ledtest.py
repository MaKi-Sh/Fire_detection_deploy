import Jetson.GPIO as GPIO

print("Initializing GPIO...")
GPIO.setmode(GPIO.BOARD)

print("Setting up pin 7 as output...")
GPIO.setup(7, GPIO.OUT)

print("Turning LED ON (pin 7 set to HIGH)...")
GPIO.output(7, GPIO.HIGH)

input("LED is on. Press Enter to turn off...")

print("Turning LED OFF (pin 7 set to LOW)...")
GPIO.output(7, GPIO.LOW)

print("Cleaning up GPIO...")
GPIO.cleanup()
print("Done!")
