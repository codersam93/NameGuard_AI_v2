from fastapi import FastAPI, APIRouter, HTTPException, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from datetime import datetime, timezone

try:
    from backend.models import EvaluateNamesRequest, EvaluateNamesResponse
    from backend.scoring import evaluate_names
except ImportError:
    from models import EvaluateNamesRequest, EvaluateNamesResponse
    from scoring import evaluate_names


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# MongoDB connection (retained primarily for health & future persistence)
mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get("DB_NAME", "test_database")]

# Create the main app without a prefix
app = FastAPI(title="MCA Name Acceptance Probability API")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.get("/")
async def root_check():
    return {"message": "Root route hit"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/healthz")
async def healthz_check():
    return {"status": "ok"}

# Create a router with the /api prefix
api_router = APIRouter()


# Middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        raise


# Health route
@api_router.get("/")
async def root():
    return {"message": "MCA Name Acceptance Probability API is running"}


# ---------------------------
# New MCA name evaluation endpoint
# ---------------------------


@api_router.post("/evaluate-names", response_model=EvaluateNamesResponse)
async def evaluate_names_endpoint(payload: EvaluateNamesRequest) -> EvaluateNamesResponse:
    logger.info(f"Evaluating names: {payload.names}")
    if not payload.names:
        raise HTTPException(status_code=400, detail="At least one name is required")

    # Simple safeguard against abuse
    if len(payload.names) > 3:
        raise HTTPException(status_code=400, detail="A maximum of 3 names can be evaluated in one request")

    try:
        results = evaluate_names(payload.names)
        return EvaluateNamesResponse(results=results, evaluated_at=datetime.now(timezone.utc))
    except Exception as e:
        logger.error(f"Error evaluating names: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include the router in the main app
app.include_router(api_router, prefix="/api")
app.include_router(api_router)

# CORS Configuration
# allow_credentials=False is safer when allow_origins=["*"]
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
