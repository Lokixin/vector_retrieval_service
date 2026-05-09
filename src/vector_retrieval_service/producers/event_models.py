from pydantic import BaseModel

from vector_retrieval_service.domain.factories import LanguageModels


class TextToEncodeEvent(BaseModel):
    requested_model: LanguageModels
    text_to_encode: str
    _id: str
    requested_at: str
