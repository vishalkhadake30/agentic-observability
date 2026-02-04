"""
Configuration Settings

Centralized configuration management using Pydantic Settings
with environment variable support.
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_env: str = Field(default="development", description="Environment")
    app_host: str = Field(default="0.0.0.0", description="API host")
    app_port: int = Field(default=8000, description="API port")
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="logs", description="Log files directory")
    enable_file_logging: bool = Field(default=True, description="Enable file-based logging")
    
    # Anthropic Claude
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Claude model"
    )
    
    # HuggingFace LLM Integration
    huggingface_token: str = Field(default="", description="HuggingFace API token")
    llm_model: str = Field(
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        description="HuggingFace LLM model for reasoning agent"
    )
    llm_max_tokens: int = Field(default=500, description="Max tokens for LLM response")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    
    # TimescaleDB
    timescale_host: str = Field(default="localhost", description="TimescaleDB host")
    timescale_port: int = Field(default=5432, description="TimescaleDB port")
    timescale_database: str = Field(default="observability", description="Database name")
    timescale_user: str = Field(default="postgres", description="Database user")
    timescale_password: str = Field(default="postgres", description="Database password")
    
    @property
    def database_url(self) -> str:
        """Construct database URL"""
        return (
            f"postgresql+asyncpg://{self.timescale_user}:{self.timescale_password}"
            f"@{self.timescale_host}:{self.timescale_port}/{self.timescale_database}"
        )
    
    # OpenTelemetry
    otel_exporter_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP endpoint"
    )
    otel_service_name: str = Field(
        default="agentic-observability",
        description="Service name"
    )
    
    # Agent Configuration
    anomaly_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Anomaly detection threshold"
    )
    rag_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="Embedding model"
    )
    rag_chunk_size: int = Field(default=512, description="RAG chunk size")
    rag_overlap: int = Field(default=50, description="RAG chunk overlap")
    max_agent_retries: int = Field(default=3, description="Max retries per agent")
    circuit_breaker_threshold: int = Field(
        default=5,
        description="Circuit breaker failure threshold"
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        description="Circuit breaker timeout (seconds)"
    )
    
    # Vector Database
    vector_db_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant URL"
    )
    vector_collection_name: str = Field(
        default="observability_embeddings",
        description="Vector collection"
    )
    
    # Redis
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")


# Global settings instance
settings = Settings()
