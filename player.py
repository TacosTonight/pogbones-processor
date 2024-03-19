import pyautogui
import time
import os
import json
import pyautogui
from confluent_kafka import Consumer
from dotenv import load_dotenv


def press_button(direction_mapping, button):
    pyautogui.keyDown(direction_mapping.get("{}".format(button)))
    time.sleep(0.5)
    pyautogui.keyUp(direction_mapping.get("{}".format(button)))


if __name__ == "__main__":
    load_dotenv()
    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVER"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "group.id": "emulator",
        "sasl.username": os.getenv("KAFKA_KEY"),
        "sasl.password": os.getenv("KAFKA_SECRET"),
        "auto.offset.reset": "latest",
    }

    direction_mapping = {
        "up": "up",
        "down": "down",
        "left": "left",
        "right": "right",
        "A": "x",
        "B": "z",
    }
    x, y = (1180, 200)

    # Move mouse to XY coordinates
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(0.5)

    consumer = Consumer(conf)
    try:
        consumer.subscribe([os.getenv("KAFKA_TOPIC_BUTTON_NAME")])

        while True:
            message = consumer.poll(1.0)

            if message is None:
                continue
            if message.error():
                print(f"Consumer error: {message.error()}")
                continue

            print(f"Received message: {message.value().decode('utf-8')}")
            message = message.value().decode("utf-8")
            button = json.loads(message).get("button")
            press_button(direction_mapping, button)
            print(f"Button Pressed: {button}")
    finally:
        consumer.close()
