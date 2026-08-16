"""Stage 3 RPC security and paper-safe contract checks against PostgreSQL."""

from __future__ import annotations

import json
import os
import uuid

import asyncpg
import pytest

SCOPE = ("stage3-board", "stage3-class", "stage3-subject")
SPEC = {
    "duration_minutes": 60,
    "sections": [
        {
            "key": "A",
            "title": "MCQ",
            "type": "mcq",
            "select_count": 1,
            "attempt_count": 1,
            "marks_each": 1,
            "chapter_distribution": {},
            "difficulty_distribution": {},
        }
    ],
}


@pytest.fixture
async def conn():
    try:
        connection = await asyncpg.connect(
            os.getenv(
                "TEST_DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/taleem_dev",
            )
        )
    except (ConnectionRefusedError, OSError):
        pytest.skip("PostgreSQL database is unavailable")
    transaction = connection.transaction()
    await transaction.start()
    try:
        yield connection
    finally:
        await transaction.rollback()
        await connection.close()


async def add_approved_mcq(conn: asyncpg.Connection) -> str:
    question_id = await conn.fetchval(
        "INSERT INTO question_bank_questions(source,created_by) VALUES('stage3-test','test') RETURNING id"
    )
    text = f"Stage 3 MCQ {uuid.uuid4()}"
    revision_id = await conn.fetchval(
        """INSERT INTO question_bank_revisions(question_id,version_no,board_id,class_id,subject_id,chapter_id,answer_mode,difficulty,marks,question_text,normalized_question,question_hash,answer_blocks,review_status,source,approved_by,approved_at,created_by)
        VALUES($1,1,$2,$3,$4,'chapter-1','mcq','easy',1,$5,$5,$6,'[{"type":"paragraph","text":"hidden answer"}]','approved','stage3-test','test',NOW(),'test') RETURNING id""",
        question_id,
        *SCOPE,
        text,
        "b" * 64,
    )
    await conn.execute(
        "INSERT INTO question_bank_mcq_options(revision_id,option_key,option_text,display_order,is_correct) VALUES($1,'A','Safe option',0,TRUE)",
        revision_id,
    )
    return str(question_id)


async def generate(conn: asyncpg.Connection, mode: str, spec: dict | None = None):
    raw = await conn.fetchval(
        "SELECT taleem_generate_test_paper($1,$2,$3,$4,$5::jsonb,$6)",
        mode,
        *SCOPE,
        json.dumps(spec) if spec else None,
        "stage3-seed",
    )
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.mark.asyncio
async def test_board_generation_is_ephemeral_and_hides_answers(conn):
    question_id = await add_approved_mcq(conn)
    await conn.execute(
        "INSERT INTO board_paper_blueprints(board_id,class_id,subject_id,name,config,is_active,created_by,updated_by) VALUES($1,$2,$3,'Stage 3',$4::jsonb,TRUE,'test','test')",
        *SCOPE,
        json.dumps(SPEC),
    )
    paper = await generate(conn, "board")
    question = paper["sections"][0]["questions"][0]
    assert question["id"] == question_id
    assert question["options"] == [{"key": "A", "text": "Safe option"}]
    serialized = json.dumps(paper)
    assert (
        "is_correct" not in serialized
        and "answer_blocks" not in serialized
        and "hidden answer" not in serialized
    )


@pytest.mark.asyncio
async def test_custom_validation_and_direct_permissions_are_enforced(conn):
    await add_approved_mcq(conn)
    paper = await generate(conn, "custom", SPEC)
    assert paper["mode"] == "custom"
    with pytest.raises(asyncpg.RaiseError, match="INVALID_CUSTOM_SPEC"):
        async with conn.transaction():
            await generate(conn, "custom", {"duration_minutes": 0, "sections": []})
    signature = "taleem_generate_test_paper(text,text,text,text,jsonb,text)"
    assert not await conn.fetchval(
        "SELECT has_function_privilege('anon', $1, 'EXECUTE')", signature
    )
    assert not await conn.fetchval(
        "SELECT has_function_privilege('authenticated', $1, 'EXECUTE')", signature
    )
    assert await conn.fetchval(
        "SELECT has_function_privilege('service_role', $1, 'EXECUTE')", signature
    )
    selector = "taleem_select_questions(text,text,text,jsonb,text)"
    assert not await conn.fetchval(
        "SELECT has_function_privilege('anon', $1, 'EXECUTE')", selector
    )
    assert not await conn.fetchval(
        "SELECT has_function_privilege('authenticated', $1, 'EXECUTE')", selector
    )
