"""Provider registry and factory for ModelProvider implementations."""

from . import qwen_provider

# Mapping of provider name to provider class
_PROVIDERS = {
    "qwen": qwen_provider.QwenProvider,
    # Additional providers (e.g., claude, openrouter) can be added here
}

def get_provider(provider_name: str, **kwargs):
    """Factory function to instantiate a ModelProvider by name."""
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown provider: {provider_name}")
    return _PROVIDERS[provider_name](**kwargs)