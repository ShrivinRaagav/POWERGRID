from typing import Dict, Type, List
from src.models.base_model import BaseForecastModel
from src.utils.helpers import setup_logger

logger = setup_logger("model_registry")

# Central repository of forecasting models
_MODEL_REGISTRY: Dict[str, Type[BaseForecastModel]] = {}

def register_model(name: str):
    """
    Decorator to register a forecasting model class in the registry.
    
    Example:
    @register_model("random_forest")
    class RandomForestModel(BaseForecastModel):
        ...
    """
    def decorator(cls: Type[BaseForecastModel]):
        if not issubclass(cls, BaseForecastModel):
            raise TypeError(f"Class '{cls.__name__}' must inherit from BaseForecastModel to be registered.")
            
        cleaned_name = name.strip().lower()
        if cleaned_name in _MODEL_REGISTRY:
            logger.warning(f"Overwriting registered model '{cleaned_name}' with {cls.__name__}.")
            
        _MODEL_REGISTRY[cleaned_name] = cls
        logger.info(f"Successfully registered forecasting model '{cleaned_name}' ({cls.__name__}).")
        return cls
    return decorator

def get_model_class(name: str) -> Type[BaseForecastModel]:
    """Retrieves a registered forecasting model class by name (case-insensitive)."""
    cleaned_name = name.strip().lower()
    if cleaned_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Forecasting model '{cleaned_name}' not found in registry. "
            f"Registered models: {list_registered_models()}"
        )
    return _MODEL_REGISTRY[cleaned_name]

def list_registered_models() -> List[str]:
    """Returns a list of all registered case-insensitive model names."""
    return list(_MODEL_REGISTRY.keys())
