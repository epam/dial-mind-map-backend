from enum import Enum


class EdgeType(str, Enum):
    GENERATED = "Generated"
    MANUAL = "Manual"
    INIT = "Init"
