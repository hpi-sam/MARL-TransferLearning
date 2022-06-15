import enum
from typing import List

class ComponentFailure(enum.Enum):
    NONE = 0
    CF1 = 1
    CF2 = 2
    CF3 = 3
    CF5 = 4

    @classmethod
    def list(self) -> List[str]:
        return [status.value for status in ComponentFailure]