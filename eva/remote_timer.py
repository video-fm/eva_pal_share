import time

# Timer web interface at https://shen.nz/remote_timer.html
# ABLY_API_KEY = "ykOLYw.X3XWcA:vbocFRKb_7yQAlgGNCSkxq9OYWBw6cbK_tist-FZiRo"
ABLY_UPENN_KEY = "qrc8EQ.GlGIvw:6DUKupDKrz6Y4BOij9MalhevURf9XrAiY32cxBEgJ4w"

class RemoteTimer:
    def __init__(self):
        try:
            from ably.sync import AblyRestSync
        except ImportError as e:
            raise ImportError("Missing ably. You can run `pip install ably`") from e

        self.ably = AblyRestSync(ABLY_UPENN_KEY)
        self.channel = self.ably.channels.get("timer-control")

    def reset(self):
        self.channel.publish("command", {"command": "reset", "status": ""})

    def toggle(self, status: str):
        self.channel.publish("command", {"command": "toggle", "status": status})

    def set_status(self, status: str):
        self.channel.publish("command", {"command": "noop", "status": status})


if __name__ == "__main__":
    remote_timer = RemoteTimer()
    remote_timer.reset()
    time.sleep(1)
    remote_timer.toggle("on")
    time.sleep(1)
    remote_timer.set_status("running")
    time.sleep(1)
    remote_timer.toggle("off")