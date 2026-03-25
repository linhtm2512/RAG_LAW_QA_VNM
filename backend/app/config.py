from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    collection_name: str = "vnqa_documents"

    # Embedding model
    # Options:
    #   "keepitreal/vietnamese-sbert"              – Vietnamese Sentence-BERT (recommended)
    #   "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base" – Vietnamese PhoBERT SimCSE
    #   "intfloat/multilingual-e5-large"           – Multilingual E5 (high quality)
    embedding_model: str = "keepitreal/vietnamese-sbert"
    embedding_dim: int = 768  # auto-detected at startup; overridden if needed

    # LLM
    llm_provider: str = "glm"  # "gemini" | "openai" | "glm"
    gemini_api_key: str = ""
    openai_api_key: str = ""
    llm_model_gemini: str = "gemini-2.5-flash"
    llm_model_openai: str = "gpt-4o-mini"
    # GLM / VNPay AI Gateway
    glm_api_key: str = ""
    glm_base_url: str = "https://genai.vnpay.vn/aigateway/llm_glm47/v1/"
    llm_model_glm: str = "glm-4.7-flash"

    # Documents
    documents_path: str = "./documents"

    # Chunking
    chunk_size: int = 400  # words per chunk
    chunk_overlap: int = 50  # words overlap between chunks

    # Retrieval
    top_k: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
