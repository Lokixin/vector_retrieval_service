from typing import Any

import asyncer
from PIL import Image

from vector_retrieval_service.config import DEVICE
from vector_retrieval_service.domain import factories


async def get_image_embeddings_services(
    image: Image.Image, requested_model: str
) -> dict[str, Any]:
    pre_processor, model = factories.make_image_model(requested_model)
    input_to_model = pre_processor(images=image, return_tensors="pt").to(DEVICE)
    model_output = await asyncer.asyncify(model.get_image_features)(**input_to_model)
    return {
        "model_used": "CLIP",
        "embeddings": model_output.detach().cpu().numpy().tolist(),
    }
