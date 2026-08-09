from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

class ModelProvider(ABC):
    """Abstract interface for model providers."""

    @abstractmethod
    def generate(self, prompt: str, context: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate a response given a prompt and optional context."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return provider and model identification metadata."""
        pass