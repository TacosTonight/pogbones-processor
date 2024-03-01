from twitchio.ext import commands


class TwitchBot(commands.Bot):
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
        await self.message_queue.put(message)
