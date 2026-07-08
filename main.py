import logging
import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipeline import run_pipeline, validate_api_key, PatientInput, StewardshipReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PCT Antibiotic Stewardship API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class ApiKeyRequest(BaseModel):
    api_key: str


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/api/validate-key")
async def validate_key(request: ApiKeyRequest):
    try:
        detected = await validate_api_key(request.api_key)
        return {"ok": True, **detected}
    except ValueError as e:
        logger.error(f"API key validation error: {e}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": str(e),
                "suggestion": "Please enter a valid Gemini, OpenAI, Claude, or OpenRouter API key"
            }
        )


@app.post("/api/analyze", response_model=StewardshipReport)
async def analyze(
    patient: PatientInput,
    x_llm_api_key: str | None = Header(default=None),
    x_gemini_api_key: str | None = Header(default=None),
):
    try:
        return await run_pipeline(patient, api_key=x_llm_api_key or x_gemini_api_key)
    except ValueError as e:
        logger.error(f"Pipeline validation error: {e}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "pipeline_error",
                "message": str(e),
                "suggestion": "Please check input data and try again"
            }
        )
    except Exception as e:
        logger.error(f"Pipeline unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "message": "Pipeline failed unexpectedly",
                "suggestion": "Please try again in a few moments"
            }
        )


@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-2.5-flash"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
