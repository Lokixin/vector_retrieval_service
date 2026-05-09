from vector_retrieval_service.domain.factories import LanguageModels
from vector_retrieval_service.producers.event_creator import make_text_to_encode_event
from vector_retrieval_service.producers.event_models import TextToEncodeEvent


def test_text_event_preserves_wire_id_alias() -> None:
    event = TextToEncodeEvent(
        _id="abc123",
        requested_model=LanguageModels.MINI_LM,
        text_to_encode="hello",
        requested_at="now",
    )

    assert event.event_id == "abc123"
    assert event.model_dump(by_alias=True)["_id"] == "abc123"
    assert "_id" not in event.model_dump()


def test_created_text_event_serializes_id_for_wire_payload() -> None:
    event = make_text_to_encode_event("hello", LanguageModels.MINI_LM)

    payload = event.model_dump(by_alias=True)

    assert payload["_id"]
    assert payload["requested_model"] == LanguageModels.MINI_LM
