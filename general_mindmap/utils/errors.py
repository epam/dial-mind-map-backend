import json
import re

from openai import RateLimitError


def pretify_rate_limit(e: RateLimitError) -> str:
    dial_rate_limit = json.loads(e.response.text)

    if "display_message" in dial_rate_limit["error"]:
        return dial_rate_limit["error"]["display_message"]
    else:
        limits = re.findall(
            r"(\d+) / (\d+)", dial_rate_limit["error"]["message"]
        )

        limits = [(min(limit[0], limit[1]), limit[1]) for limit in limits]

        if limits[3][0] >= limits[3][1]:
            return f"You've used all your monthly tokens ({limits[3][0]} out of {limits[3][1]})."
        elif limits[2][0] >= limits[2][1]:
            return f"You've used all your weekly tokens ({limits[2][0]} out of {limits[2][1]})."
        elif limits[1][0] >= limits[1][1]:
            return f"You've used all your daily tokens ({limits[1][0]} out of {limits[1][1]})."
        else:
            return f"You've used all your minute tokens ({limits[0][0]} out of {limits[0][1]})."
