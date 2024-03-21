"""
This file was created to demonstrate how to use the TwitchIO library to create 
a bot that can connect to Twitch chat and run sentiment analysis without using 
DE technologies like Kafka or Spark Structured Streaming that I used in 
twitch_producer.py and twitch_consumer.py.

This was essentially a proof of concept so I could see how Kafka and Spark make 
the process easier and more scalable when I moved to the next step of the project.
"""

import asyncio
import time
import heapq
import os
from twitchio.ext import commands
from transformers import pipeline
from dotenv import load_dotenv
from datetime import datetime
from twitch_client import TwitchClient
from processed_message import ProcessedMessage


class Bot(commands.Bot):
    def __init__(self, token, prefix, initial_channels, message_queue):
        super().__init__(token=token, prefix=prefix, initial_channels=initial_channels)
        self.message_queue = message_queue

    async def event_ready(self):
        # Notify us when everything is ready!
        # We are logged in and ready to chat and use commands...
        print(f"Logged in as | {self.nick}")
        print(f"User id is | {self.user_id}")

    async def event_message(self, message):
        # Messages with echo set to True are messages sent by the bot...
        # For now we just want to ignore them...
        if message.echo:
            return

        # Print the contents of our message to console...
        print(
            f"IN: CHAT_TIME_STAMP: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')} "
            f"CURRENT_TIME_STAMP: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} "
            f"AUTHOR: {message.author} "
            f"CONTENT: {message.content} "
            f"ID: {message.id}"
        )

        await self.message_queue.put(message)


async def write_results_to_heap(output_heap, output_lock, message):
    async with output_lock:
        heapq.heappush(output_heap, message)


async def classify_message(
    classifier,
    message_queue,
    output_heap,
    output_lock,
):
    while True:
        message = await message_queue.get()
        s = time.time()
        result = classifier(message.content)
        e = time.time()
        processed_message = ProcessedMessage(message, sentiment=result[0])
        await write_results_to_heap(output_heap, output_lock, processed_message)


async def output_results(output_heap, output_lock):
    while True:
        async with output_lock:
            if output_heap:
                message = heapq.heappop(output_heap)
                print(
                    f"OUT: CHAT_TIME_STAMP: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')} "
                    f"CURRENT_TIME_STAMP: {datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S.%f')} "
                    f"AUTHOR: {message.author} "
                    f"CONTENT: {message.content} "
                    f"ID: {message.id} "
                    f"SENTIMENT: {message.sentiment}"
                )
        await asyncio.sleep(0.1)


async def main():
    # Create a queue to pass messages between the bot and the main thread...
    output_heap = []
    output_lock = asyncio.Lock()
    message_queue = asyncio.Queue(maxsize=100)
    classifier = pipeline(
        "sentiment-analysis",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
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

    num_workers = 100
    workers = [
        asyncio.create_task(
            classify_message(
                classifier=classifier,
                message_queue=message_queue,
                output_heap=output_heap,
                output_lock=output_lock,
            )
        )
        for _ in range(num_workers)
    ]

    await asyncio.gather(
        bot.start(),
        *workers,
        output_results(output_heap, output_lock),
    )


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
