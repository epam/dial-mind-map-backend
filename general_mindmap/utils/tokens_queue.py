import re
from typing import Dict

MAX_QUEUE_SIZE = 10


def next_interesting_char(a: str):
    p1 = a.find("[")
    p2 = a.find("{")

    if p1 == -1:
        return p2
    elif p2 == -1:
        return p1
    else:
        return min(p1, p2)


class TokensQueue:
    tokens: str = ""
    nodes_map: Dict[str, str]
    chunks_map: Dict[str, str]

    def __init__(self, nodes_map: Dict[str, str], chunks_map: Dict[str, str]):
        self.nodes_map = nodes_map
        self.chunks_map = chunks_map

    def add(self, token: str) -> str:
        tokens = self.tokens + token

        result = ""
        while len(tokens) > 0:
            if tokens[0] != "[" and tokens[0] != "{":
                result += tokens[0]
                tokens = tokens[1:]
                continue

            if tokens[0] == "[":
                end = tokens.find("]")

                if end == -1:
                    if len(tokens) >= MAX_QUEUE_SIZE:
                        cut_pos = next_interesting_char(tokens[1:])

                        if cut_pos == -1:
                            result += tokens
                            tokens = ""
                        else:
                            result += tokens[:cut_pos]
                            tokens = tokens[cut_pos:]
                    else:
                        break  # will wait for next chars
                else:
                    if re.match(r"\[[0-9]+\]$", tokens[: end + 1]):
                        id = tokens[1:end]

                        if id in self.chunks_map:
                            result += f"^[{self.chunks_map[id]}]^"
                        else:
                            result += tokens[: end + 1]

                        tokens = tokens[end + 1 :]
                    else:
                        result += tokens[: end + 1]
                        tokens = tokens[end + 1 :]
            else:
                end = tokens.find("}")

                if end == -1:
                    if len(tokens) >= MAX_QUEUE_SIZE:
                        cut_pos = next_interesting_char(tokens[1:])

                        if cut_pos == -1:
                            result += tokens
                            tokens = ""
                        else:
                            result += tokens[:cut_pos]
                            tokens = tokens[cut_pos:]
                    else:
                        break  # will wait for next chars
                else:
                    if re.match(r"\{[0-9]+\}$", tokens[: end + 1]):
                        id = tokens[1:end]

                        if id in self.nodes_map:
                            result += f"^[{self.nodes_map[id]}]^"
                        else:
                            result += tokens[: end + 1]

                        tokens = tokens[end + 1 :]
                    else:
                        result += tokens[: end + 1]
                        tokens = tokens[end + 1 :]

        self.tokens = tokens
        return result
