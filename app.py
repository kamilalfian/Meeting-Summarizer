from fastapi import APIRouter, FastAPI
from textSummarizer.entity import SummarizerSingleRequest, SummarizerSingleResponse, SummarizerBatchRequest, SummarizerBatchResponse
from textSummarizer.pipeline.prediction import PredictionPipeline
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response

router = APIRouter(prefix="/summary", tags=["Summary"])

@router.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@router.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")

@router.post(
    "/single",
    response_model=SummarizerSingleResponse,
    name="POST single summary",
)
def single_summary(request: SummarizerSingleRequest):
    summarize = PredictionPipeline()
    return summarize.get_single_prediction(feature=request)

@router.post(
    "/batch",
    response_model=SummarizerBatchResponse,
    name="POST batch summary",
)
def batch_summary(request: SummarizerBatchRequest):
    summarize = PredictionPipeline()
    return summarize.get_batch_prediction(features=request.features)

def get_app() -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app

app = get_app()

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)