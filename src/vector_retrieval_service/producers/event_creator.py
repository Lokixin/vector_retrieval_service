import datetime
import uuid

from vector_retrieval_service.domain.factories import LanguageModels
from vector_retrieval_service.producers.event_models import TextToEncodeEvent


def make_text_to_encode_event(
    text_to_encode: str, requested_model: LanguageModels | None = None
) -> TextToEncodeEvent:
    return TextToEncodeEvent(
        text_to_encode=text_to_encode,
        requested_model=requested_model.value
        if requested_model
        else LanguageModels.MINI_LM.value,
        _id=uuid.uuid4().hex,
        requested_at=str(datetime.datetime.now(tz=datetime.timezone.utc)),
    )
