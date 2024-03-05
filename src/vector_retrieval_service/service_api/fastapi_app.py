import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from vector_retrieval_service.service_api.views.iamge_views import image_router
from vector_retrieval_service.service_api.views.multimodal_services import (
    multimodal_router,
)
from vector_retrieval_service.service_api.views.text_views import retrieval_service


logging.basicConfig(level=logging.INFO)
app = FastAPI()
app.include_router(retrieval_service)
app.include_router(image_router)
app.include_router(multimodal_router)


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"message": "The app is up and running!"}


@app.get("/")
def redirect_to_docs() -> RedirectResponse:
    return RedirectResponse(url="docs", status_code=302)
