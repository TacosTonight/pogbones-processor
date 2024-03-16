import asyncio
import os
import socket
from twitch_bot import TwitchBot
from twitch_client import TwitchClient
from confluent_kafka import Producer, KafkaException
from dotenv import load_dotenv


def delivery_callback(err, msg):
    if err:
        print("ERROR: Message failed delivery: {}".format(err))
    else:
        print(
            "Produced event to topic {topic}: key = {key:12} value = {value:12}".format(
                topic=msg.topic(),
                key=msg.key().decode("utf-8"),
                value=msg.value().decode("utf-8"),
            )
        )


def prepare_message(message):
    try:
        channel = message.channel.name
        value = message.content
        timestamp = int(message.timestamp.timestamp() * 1000)
        key = f"{channel}-{timestamp}"
        return {"key": key, "value": value, "timestamp": timestamp, "channel": channel}
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except KeyError as e:
        print(f"KeyError: {e}")


async def write_results_to_kafka(producer, topic, message_queue):
    try:
        while True:
            message = await message_queue.get()
            prepared_message = prepare_message(message)
            producer.produce(
                topic,
                key=prepared_message.get("key"),
                value=prepared_message.get("value"),
                timestamp=prepared_message.get("timestamp"),
                callback=delivery_callback,
            )
            producer.poll(0)
    except AttributeError as e:
        print(f"AttributeError: {e}")
    except KafkaException as e:
        print(f"KafkaException: {e}")


async def main():
    message_queue = asyncio.Queue(maxsize=100)
    twitch_client = TwitchClient()
    twitch_streamers = twitch_client.extract_streamer_usernames(
        twitch_client.get_streams()
    )

    # Create the bot...
    bot = TwitchBot(
        token=os.getenv("TWITCH_USER_TOKEN"),
        prefix="!",
        initial_channels=["parkenharbor"],
        message_queue=message_queue,
    )

    conf = {
        "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVER"),
        "security.protocol": "SASL_SSL",
        "sasl.mechanism": "PLAIN",
        "sasl.username": os.getenv("KAFKA_KEY"),
        "sasl.password": os.getenv("KAFKA_SECRET"),
        "client.id": socket.gethostname(),
    }
    producer = Producer(conf)

    num_workers = 10
    workers = [
        asyncio.create_task(
            write_results_to_kafka(
                producer=producer,
                topic=os.getenv("KAFKA_TOPIC_NAME"),
                message_queue=message_queue,
            )
        )
        for _ in range(num_workers)
    ]

    await asyncio.gather(bot.start(), *workers)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
