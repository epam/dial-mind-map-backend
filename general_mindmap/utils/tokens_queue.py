import re
from typing import Dict

MAX_QUEUE_SIZE = 10
CITATION_PATTERN = re.compile(r"^\[(\d{1,5})(\.(\d{1,5})){2}\]$")


def next_interesting_char(a: str):
    return a.find("[")


class TokensQueue:
    tokens: str = ""
    transofmation_map: Dict[str, str]
    last_char: str = ""

    def __init__(self, transofmation_map: Dict[str, str]):
        self.transofmation_map = transofmation_map

    def add(self, token: str) -> str:
        tokens = self.tokens + token

        result = ""
        while len(tokens) > 0:
            if tokens[0] != "[":
                result += tokens[0]
                tokens = tokens[1:]
                continue

            end = tokens.find("]")

            if end == -1:
                if len(tokens) >= MAX_QUEUE_SIZE:
                    cut_pos = next_interesting_char(tokens[1:])

                    if cut_pos == -1:
                        result += tokens
                        tokens = ""
                    else:
                        cut_pos += 1

                        result += tokens[:cut_pos]
                        tokens = tokens[cut_pos:]
                else:
                    break  # will wait for next token
            else:
                if re.match(r"\[[0-9]+\]$", tokens[: end + 1]):
                    id = tokens[1:end]

                    if id in self.transofmation_map:
                        result += f"^[{self.transofmation_map[id]}]^"
                    else:
                        result += tokens[: end + 1]

                    tokens = tokens[end + 1 :]
                else:
                    if (
                        CITATION_PATTERN.match(tokens[: end + 1])
                        and self.last_char != "^"
                    ):
                        result += f"^{tokens[: end + 1]}^"
                    else:
                        result += tokens[: end + 1]

                    tokens = tokens[end + 1 :]

        self.tokens = tokens

        if result:
            self.last_char = result[-1]

        return result
