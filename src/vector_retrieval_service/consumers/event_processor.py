import logging
from typing import Any

from vector_retrieval_service.domain.ai_models import LanguageModelInterface
from vector_retrieval_service.domain.factories import LanguageModels


logger = logging.getLogger(__name__)


class HeaderException(Exception):
    ...


class UnprocessableEvent(Exception):
    pass


def _decode_headers(headers: list[tuple[str, bytes]]) -> str:
    for header in headers:
        if not header[0] == "event_type":
            continue
        else:
            return header[1].decode("utf-8")
    raise HeaderException("Missing event_type header")


async def process_event(event: Any) -> Any:
    header = _decode_headers(event.headers)
    match header:
        case "text":
            return await process_text_event(event.value)
        case _:
            raise UnprocessableEvent(f"Unknown event_type: {header}")


async def process_text_event(event_value: dict[str, Any]) -> Any:
    try:
        requested_model_value = event_value["requested_model"]
        text_to_encode = event_value["text_to_encode"]
        requested_model = LanguageModels(requested_model_value)
        model_interface = LanguageModelInterface(language_model=requested_model)
        embeddings = await model_interface.encode(model_input=text_to_encode)
        return embeddings
    except ValueError as e:
        logger.error("The requested model is not valid")
        raise e
    except KeyError as e:
        logger.error("Model not specified")
        raise e
