import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

from backend.api import router
from backend.embedding_utils import get_embedding_model
from backend.llm_provider import get_llm_provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load and cache lightweight embedding model and active LLM provider
    get_embedding_model()
    get_llm_provider()
    yield


app = FastAPI(
    title="IntelliRAG — Document-Based Multi-Hop RAG",
    description="Multi-hop document QA with hybrid FAISS+BM25 retrieval, PRM, PEM, and Gemini Flash LLM.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
