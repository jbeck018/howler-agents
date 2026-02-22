"""Service configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://howler:howler@localhost:5432/howler_agents"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Service ports
    grpc_port: int = 50051
    rest_port: int = 8080

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Evolution defaults
    howler_population_size: int = 10
    howler_group_size: int = 3
    howler_num_iterations: int = 5
    howler_alpha: float = 0.5
    howler_num_probes: int = 20

    # LLM
    howler_llm_acting_model: str = "claude-sonnet-4-20250514"
    howler_llm_evolving_model: str = "claude-sonnet-4-20250514"
    howler_llm_reflecting_model: str = "claude-sonnet-4-20250514"

    # JWT Auth
    jwt_secret: str = "change-me-in-production-use-a-long-random-string"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
