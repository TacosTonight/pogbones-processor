import requests
import os


class TwitchClient:
    def __init__(self):
        self.access_token = self.get_access_token()

    def get_access_token(self):
        url = "https://id.twitch.tv/oauth2/token"
        data = {
            "client_id": os.getenv("TWITCH_CLIENT_ID"),
            "client_secret": os.getenv("TWITCH_CLIENT_SECRET"),
            "grant_type": "client_credentials",
        }
        r = requests.post(url=url, data=data)
        return r.json()["access_token"]

    def get_streams(self):
        url = "https://api.twitch.tv/helix/streams"
        r = requests.get(
            url=url,
            headers={
                "Client-id": os.getenv("TWITCH_CLIENT_ID"),
                "Authorization": f"Bearer {self.access_token}",
            },
        )
        return r.json()

    def extract_streamer_usernames(self, stream_data):
        usernames = []
        for stream in stream_data["data"]:
            usernames.append(stream["user_name"])
        return usernames
