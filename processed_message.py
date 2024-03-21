"""
Used in bot.py to create a ProcessedMessage object that extends the 
Message object from twitchio to add sentiment analysis.
"""

from twitchio import Message


class ProcessedMessage(Message):
    def __init__(self, current_message, sentiment, **kwargs):
        author = current_message.author
        channel = current_message.channel
        content = current_message.content
        echo = current_message.echo
        first = current_message.first
        hype_chat_data = current_message.hype_chat_data
        id = current_message.id
        raw_data = current_message.raw_data
        tags = current_message.tags
        timestamp = current_message.timestamp
        super().__init__(
            author=author,
            channel=channel,
            content=content,
            echo=echo,
            first=first,
            hype_chat_data=hype_chat_data,
            id=id,
            raw_data=raw_data,
            tags=tags,
            timestamp=timestamp,
            **kwargs,
        )
        self.sentiment = sentiment

    def __lt__(self, other):
        return self.timestamp < other.timestamp
