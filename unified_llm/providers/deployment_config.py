"""
Deployment Configuration Layer
==============================

Configuration management for hosting capabilities and feature flags.

Features:
- Feature flag management for Novita-style capabilities
- Hosting provider configuration
- API endpoint management
- Resource pooling configuration
- Environment-specific settings

(c) Helix Collective 2024 - Proprietary Technology Stack
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class HostingProvider(Enum):
    """Supported hosting providers"""

    NOVITA = "novita"
    RUNPOD = "runpod"
    VAST_AI = "vast.ai"
    RAILWAY = "railway"
    LOCAL = "local"
    CUSTOM = "custom"


class GPUProvider(Enum):
    """Supported GPU providers"""

    NOVITA = "novita"
    RUNPOD = "runpod"
    VAST_AI = "vast.ai"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


@dataclass
class DeploymentConfig:
    """Deployment configuration for hosting capabilities"""

    # Core LLM features (currently ON)
    llm_inference_enabled: bool = True
    coordination_integration: bool = True

    # Future hosting features (currently OFF)
    model_api_enabled: bool = False  # REST API for external requests
    gpu_cloud_enabled: bool = False  # GPU resource pooling
    agent_sandbox_enabled: bool = False  # Isolated agent environments
    custom_training_enabled: bool = False  # User-provided datasets

    # Hosting provider (for future)
    hosting_provider: HostingProvider | None = None
    gpu_provider: GPUProvider | None = None

    # API configuration
    api_rate_limit: int = 100  # requests/minute
    api_port: int = 8000
    api_host: str = "0.0.0.0"  # nosec B104
    api_cors_origins: list[str] = field(default_factory=list)  # Empty = no CORS; set explicitly per environment

    # GPU resource configuration
    gpu_pool_size: int = 1
    gpu_memory_per_instance: int = 24  # GB
    max_concurrent_requests: int = 10
    gpu_autoscaling_enabled: bool = False

    # Agent sandbox configuration
    sandbox_isolation: bool = True
    sandbox_memory_limit: int = 8  # GB
    sandbox_cpu_limit: int = 4  # cores
    sandbox_network_isolation: bool = True

    # Custom training configuration
    training_dataset_path: str | None = None
    training_batch_size: int = 32
    training_epochs: int = 10
    training_learning_rate: float = 1e-4

    # Environment settings
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False
    log_level: str = "INFO"

    # Security settings
    api_key_required: bool = False
    jwt_secret: str | None = None
    allowed_ips: list[str] = field(default_factory=list)

    # Monitoring and observability
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    health_check_enabled: bool = True

    @classmethod
    def from_env(cls) -> "DeploymentConfig":
        """Load configuration from environment variables"""
        config = cls()

        # Feature flags
        config.llm_inference_enabled = os.getenv("LLM_INFERENCE_ENABLED", "true").lower() == "true"
        config.coordination_integration = os.getenv("COORDINATION_INTEGRATION", "true").lower() == "true"
        config.model_api_enabled = os.getenv("MODEL_API_ENABLED", "false").lower() == "true"
        config.gpu_cloud_enabled = os.getenv("GPU_CLOUD_ENABLED", "false").lower() == "true"
        config.agent_sandbox_enabled = os.getenv("AGENT_SANDBOX_ENABLED", "false").lower() == "true"
        config.custom_training_enabled = os.getenv("CUSTOM_TRAINING_ENABLED", "false").lower() == "true"

        # Hosting provider
        hosting_provider = os.getenv("HOSTING_PROVIDER")
        if hosting_provider:
            try:
                config.hosting_provider = HostingProvider(hosting_provider)
            except ValueError:
                logger.warning("Invalid hosting provider: %s", hosting_provider)

        gpu_provider = os.getenv("GPU_PROVIDER")
        if gpu_provider:
            try:
                config.gpu_provider = GPUProvider(gpu_provider)
            except ValueError:
                logger.warning("Invalid GPU provider: %s", gpu_provider)

        # API configuration
        config.api_rate_limit = int(os.getenv("API_RATE_LIMIT", "100"))
        config.api_port = int(os.getenv("API_PORT", "8000"))
        config.api_host = os.getenv("API_HOST", "0.0.0.0")  # nosec B104
        cors_origins_env = os.getenv("API_CORS_ORIGINS", "")
        config.api_cors_origins = (
            [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else []
        )

        # GPU configuration
        config.gpu_pool_size = int(os.getenv("GPU_POOL_SIZE", "1"))
        config.gpu_memory_per_instance = int(os.getenv("GPU_MEMORY_PER_INSTANCE", "24"))
        config.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
        config.gpu_autoscaling_enabled = os.getenv("GPU_AUTOSCALING_ENABLED", "false").lower() == "true"

        # Agent sandbox configuration
        config.sandbox_isolation = os.getenv("SANDBOX_ISOLATION", "true").lower() == "true"
        config.sandbox_memory_limit = int(os.getenv("SANDBOX_MEMORY_LIMIT", "8"))
        config.sandbox_cpu_limit = int(os.getenv("SANDBOX_CPU_LIMIT", "4"))
        config.sandbox_network_isolation = os.getenv("SANDBOX_NETWORK_ISOLATION", "true").lower() == "true"

        # Custom training configuration
        config.training_dataset_path = os.getenv("TRAINING_DATASET_PATH")
        config.training_batch_size = int(os.getenv("TRAINING_BATCH_SIZE", "32"))
        config.training_epochs = int(os.getenv("TRAINING_EPOCHS", "10"))
        config.training_learning_rate = float(os.getenv("TRAINING_LEARNING_RATE", "1e-4"))

        # Environment settings
        config.environment = os.getenv("ENVIRONMENT", "development")
        config.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        config.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Security settings
        # CONSOLIDATED: All JWT secret retrieval uses the single canonical function
        config.api_key_required = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
        try:
            from apps.backend.core.unified_auth import _get_jwt_secret

            config.jwt_secret = _get_jwt_secret()
        except RuntimeError:
            config.jwt_secret = None
        config.allowed_ips = os.getenv("ALLOWED_IPS", "").split(",")

        # Monitoring and observability
        config.metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        config.tracing_enabled = os.getenv("TRACING_ENABLED", "false").lower() == "true"
        config.health_check_enabled = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"

        logger.info("✅ Deployment configuration loaded from environment")
        return config

    @classmethod
    def from_file(cls, config_path: str) -> "DeploymentConfig":
        """Load configuration from JSON file"""
        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)

            config = cls()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            logger.info("✅ Deployment configuration loaded from %s", config_path)
            return config
        except FileNotFoundError:
            logger.warning("Configuration file not found: %s", config_path)
            return cls()
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in configuration file: %s", e)
            return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "llm_inference_enabled": self.llm_inference_enabled,
            "coordination_integration": self.coordination_integration,
            "model_api_enabled": self.model_api_enabled,
            "gpu_cloud_enabled": self.gpu_cloud_enabled,
            "agent_sandbox_enabled": self.agent_sandbox_enabled,
            "custom_training_enabled": self.custom_training_enabled,
            "hosting_provider": (self.hosting_provider.value if self.hosting_provider else None),
            "gpu_provider": self.gpu_provider.value if self.gpu_provider else None,
            "api_rate_limit": self.api_rate_limit,
            "api_port": self.api_port,
            "api_host": self.api_host,
            "api_cors_origins": self.api_cors_origins,
            "gpu_pool_size": self.gpu_pool_size,
            "gpu_memory_per_instance": self.gpu_memory_per_instance,
            "max_concurrent_requests": self.max_concurrent_requests,
            "gpu_autoscaling_enabled": self.gpu_autoscaling_enabled,
            "sandbox_isolation": self.sandbox_isolation,
            "sandbox_memory_limit": self.sandbox_memory_limit,
            "sandbox_cpu_limit": self.sandbox_cpu_limit,
            "sandbox_network_isolation": self.sandbox_network_isolation,
            "training_dataset_path": self.training_dataset_path,
            "training_batch_size": self.training_batch_size,
            "training_epochs": self.training_epochs,
            "training_learning_rate": self.training_learning_rate,
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "log_level": self.log_level,
            "api_key_required": self.api_key_required,
            "jwt_secret": self.jwt_secret,
            "allowed_ips": self.allowed_ips,
            "metrics_enabled": self.metrics_enabled,
            "tracing_enabled": self.tracing_enabled,
            "health_check_enabled": self.health_check_enabled,
        }

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, default=str)

            logger.info("✅ Deployment configuration saved to %s", config_path)
        except Exception as e:
            logger.error("Failed to save configuration to file: %s", e)

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors"""
        errors = []

        # Validate API configuration
        if self.api_port < 1 or self.api_port > 65535:
            errors.append("API port must be between 1 and 65535")

        if self.api_rate_limit < 1:
            errors.append("API rate limit must be positive")

        # Validate GPU configuration
        if self.gpu_pool_size < 0:
            errors.append("GPU pool size must be non-negative")

        if self.gpu_memory_per_instance < 1:
            errors.append("GPU memory per instance must be positive")

        if self.max_concurrent_requests < 1:
            errors.append("Max concurrent requests must be positive")

        # Validate sandbox configuration
        if self.sandbox_memory_limit < 1:
            errors.append("Sandbox memory limit must be positive")

        if self.sandbox_cpu_limit < 1:
            errors.append("Sandbox CPU limit must be positive")

        # Validate training configuration
        if self.training_batch_size < 1:
            errors.append("Training batch size must be positive")

        if self.training_epochs < 1:
            errors.append("Training epochs must be positive")

        if self.training_learning_rate <= 0:
            errors.append("Training learning rate must be positive")

        # Validate environment
        valid_environments = ["development", "staging", "production"]
        if self.environment not in valid_environments:
            errors.append(f"Environment must be one of: {valid_environments}")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Log level must be one of: {valid_log_levels}")

        return errors

    def is_production_ready(self) -> bool:
        """Check if configuration is production-ready"""
        errors = self.validate()

        # Additional production checks
        if self.environment != "production":
            errors.append("Environment must be 'production'")

        if self.debug_mode:
            errors.append("Debug mode should be disabled in production")

        if not self.api_key_required:
            errors.append("API key authentication should be enabled in production")

        if not self.sandbox_isolation:
            errors.append("Sandbox isolation should be enabled in production")

        return len(errors) == 0

    def get_feature_status(self) -> dict[str, bool]:
        """Get status of all features"""
        return {
            "llm_inference": self.llm_inference_enabled,
            "coordination_integration": self.coordination_integration,
            "model_api": self.model_api_enabled,
            "gpu_cloud": self.gpu_cloud_enabled,
            "agent_sandbox": self.agent_sandbox_enabled,
            "custom_training": self.custom_training_enabled,
        }

    def enable_feature(self, feature: str) -> bool:
        """Enable a specific feature"""
        feature_map = {
            "model_api": "model_api_enabled",
            "gpu_cloud": "gpu_cloud_enabled",
            "agent_sandbox": "agent_sandbox_enabled",
            "custom_training": "custom_training_enabled",
        }

        if feature not in feature_map:
            logger.error("Unknown feature: %s", feature)
            return False

        setattr(self, feature_map[feature], True)
        logger.info("✅ Feature enabled: %s", feature)
        return True

    def disable_feature(self, feature: str) -> bool:
        """Disable a specific feature"""
        feature_map = {
            "model_api": "model_api_enabled",
            "gpu_cloud": "gpu_cloud_enabled",
            "agent_sandbox": "agent_sandbox_enabled",
            "custom_training": "custom_training_enabled",
        }

        if feature not in feature_map:
            logger.error("Unknown feature: %s", feature)
            return False

        setattr(self, feature_map[feature], False)
        logger.info("✅ Feature disabled: %s", feature)
        return True


class FeatureFlagManager:
    """Manager for feature flags and dynamic configuration"""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.feature_overrides: dict[str, bool] = {}

    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        # Check override first
        if feature in self.feature_overrides:
            return self.feature_overrides[feature]

        # Check configuration
        feature_map = {
            "model_api": self.config.model_api_enabled,
            "gpu_cloud": self.config.gpu_cloud_enabled,
            "agent_sandbox": self.config.agent_sandbox_enabled,
            "custom_training": self.config.custom_training_enabled,
        }

        return feature_map.get(feature, False)

    def override_feature(self, feature: str, enabled: bool):
        """Override a feature flag"""
        self.feature_overrides[feature] = enabled
        logger.info("🔧 Feature override: %s = %s", feature, enabled)

    def clear_override(self, feature: str):
        """Clear a feature override"""
        if feature in self.feature_overrides:
            del self.feature_overrides[feature]
            logger.info("🔧 Feature override cleared: %s", feature)

    def get_all_features(self) -> dict[str, bool]:
        """Get status of all features"""
        return {
            "llm_inference": self.config.llm_inference_enabled,
            "coordination_integration": self.config.coordination_integration,
            "model_api": self.is_enabled("model_api"),
            "gpu_cloud": self.is_enabled("gpu_cloud"),
            "agent_sandbox": self.is_enabled("agent_sandbox"),
            "custom_training": self.is_enabled("custom_training"),
        }


# Global configuration instance
_deployment_config: DeploymentConfig | None = None
_feature_manager: FeatureFlagManager | None = None


def get_deployment_config() -> DeploymentConfig:
    """Get the global deployment configuration"""
    global _deployment_config
    if _deployment_config is None:
        _deployment_config = DeploymentConfig.from_env()
    return _deployment_config


def get_feature_manager() -> FeatureFlagManager:
    """Get the global feature flag manager"""
    global _feature_manager
    if _feature_manager is None:
        config = get_deployment_config()
        _feature_manager = FeatureFlagManager(config)
    return _feature_manager


def initialize_deployment_config(config_source: str | None = None):
    """Initialize deployment configuration"""
    global _deployment_config, _feature_manager

    if config_source and config_source.endswith(".json"):
        _deployment_config = DeploymentConfig.from_file(config_source)
    else:
        _deployment_config = DeploymentConfig.from_env()

    _feature_manager = FeatureFlagManager(_deployment_config)

    logger.info("🚀 Deployment configuration initialized")
    return _deployment_config


__all__ = [
    "DeploymentConfig",
    "FeatureFlagManager",
    "GPUProvider",
    "HostingProvider",
    "get_deployment_config",
    "get_feature_manager",
    "initialize_deployment_config",
]
