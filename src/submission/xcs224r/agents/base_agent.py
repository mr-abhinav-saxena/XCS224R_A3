from typing import Any, Dict, List

class BaseAgent(object):
    def __init__(self, **kwargs: Any) -> None:
        super(BaseAgent, self).__init__(**kwargs)

    def train(self) -> Dict[str, Any]:
        """Return a dictionary of logging information."""
        raise NotImplementedError

    def add_to_replay_buffer(self, paths: List[Dict[str, Any]]) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Any:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError