"""Railway worker handler for durable Multiple Ask answer jobs."""

from typing import Any

from app.services.multiple_ask_answers import MultipleAskAnswerService


async def handle_multiple_ask_answer(job: dict[str, Any], conn: Any) -> dict[str, str]:
    payload = job.get("payload") or {}
    session_id, epoch = payload.get("multiple_ask_session_id"), payload.get("epoch")
    if not isinstance(session_id, str) or not isinstance(epoch, int):
        raise ValueError("MULTIPLE_ASK_ANSWER_PAYLOAD_INVALID")
    workflow_status = await MultipleAskAnswerService(conn).answer(
        session_id=session_id, epoch=epoch
    )
    return {"workflow_status": workflow_status}
