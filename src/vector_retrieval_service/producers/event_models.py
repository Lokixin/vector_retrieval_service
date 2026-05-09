from pydantic import BaseModel, Field

from vector_retrieval_service.domain.factories import LanguageModels


class TextToEncodeEvent(BaseModel):
    requested_model: LanguageModels
    text_to_encode: str
    event_id: str = Field(alias="_id")
    requested_at: str
