import pytest
from pydantic import ValidationError

from nebula.schemas.repo import RelatedFeedbackRequest


def test_related_feedback_request_accepts_valid_values():
    req = RelatedFeedbackRequest(
        candidate_repo_id=2,
        feedback="helpful",
        score_snapshot=0.91,
        model_version="v1",
    )
    assert req.feedback == "helpful"


def test_related_feedback_request_rejects_invalid_feedback():
    with pytest.raises(ValidationError):
        RelatedFeedbackRequest(candidate_repo_id=2, feedback="bad")
