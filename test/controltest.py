import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setup(12, GPIO.OUT)

soft_pwm = GPIO.PWM(12, 1)
soft_pwm.start(50)
time.sleep(2)
soft_pwm.stop()
GPIO.cleanup()