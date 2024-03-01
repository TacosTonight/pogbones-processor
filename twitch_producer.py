import asyncio
import os
from bot2 import Bot
from twitch_client import TwitchClient
from confluent_kafka import Producer
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


async def write_results_to_kafka(producer, topic, message_queue):
    try:
        while True:
            message = await message_queue.get()
            channel = message.channel.name
            value = message.content
            timestamp = int(message.timestamp.timestamp() * 1000)
            key = f"{channel}-{timestamp}"
            producer.produce(
                topic,
                key=key,
                value=value,
                timestamp=timestamp,
                callback=delivery_callback,
            )
            producer.poll(0)
    except KeyError as e:
        print(f"KeyError: {e}")
    except Exception as e:
        print(f"Exception: {e}")


async def main():
    message_queue = asyncio.Queue(maxsize=100)
    twitch_client = TwitchClient()
    twitch_streamers = twitch_client.extract_streamer_usernames(
        twitch_client.get_streams()
    )

    # Create the bot...
    bot = Bot(
        token=os.getenv("TWITCH_USER_TOKEN"),
        prefix="!",
        initial_channels=twitch_streamers,
        message_queue=message_queue,
    )

    producer = Producer({"bootstrap.servers": "localhost:"})

    num_workers = 10
    workers = [
        asyncio.create_task(
            write_results_to_kafka(
                producer=producer, topic="chat", message_queue=message_queue
            )
        )
        for _ in range(num_workers)
    ]

    await asyncio.gather(bot.start(), *workers)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
