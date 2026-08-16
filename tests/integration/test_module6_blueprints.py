"""Stage 2 blueprint selector invariants against the real PostgreSQL schema."""

from __future__ import annotations

import json
import os
import uuid

import asyncpg
import pytest

SCOPE = ("module6-board", "module6-class", "module6-subject")


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


async def add_question(
    conn: asyncpg.Connection,
    *,
    chapter: str,
    difficulty: str = "easy",
    mode: str = "short",
    marks: int = 2,
    status: str = "approved",
    superseded: bool = False,
) -> tuple[str, str]:
    question_id = await conn.fetchval(
        """INSERT INTO question_bank_questions(source,created_by)
           VALUES('module6-test','test') RETURNING id::text"""
    )
    revision_id = await conn.fetchval(
        """INSERT INTO question_bank_revisions(
             question_id,version_no,board_id,class_id,subject_id,chapter_id,
             answer_mode,difficulty,marks,question_text,normalized_question,
             question_hash,answer_blocks,review_status,source,approved_by,approved_at,
             superseded_at,created_by
           ) VALUES($1::uuid,1,$2,$3,$4,$5,$6,$7,$8,$9,$9,$10,
             '[{"type":"paragraph","text":"answer"}]'::jsonb,$11,'module6-test',
             CASE WHEN $11='approved' THEN 'test' ELSE NULL END,
             CASE WHEN $11='approved' THEN NOW() ELSE NULL END,
             CASE WHEN $12 THEN NOW() ELSE NULL END,'test') RETURNING id::text""",
        question_id,
        *SCOPE,
        chapter,
        mode,
        difficulty,
        marks,
        f"Question {uuid.uuid4()}",
        "a" * 64,
        status,
        superseded,
    )
    return question_id, revision_id


def spec(sections: list[dict]) -> dict:
    return {"duration_minutes": 120, "sections": sections}


def section(
    key: str,
    *,
    select_count: int,
    attempt_count: int | None = None,
    chapters: dict[str, int] | None = None,
    difficulties: dict[str, int] | None = None,
    mode: str = "short",
    marks: int = 2,
) -> dict:
    return {
        "key": key,
        "title": key,
        "type": mode,
        "select_count": select_count,
        "attempt_count": attempt_count or select_count,
        "marks_each": marks,
        "chapter_distribution": chapters or {},
        "difficulty_distribution": difficulties or {},
    }


async def select(conn: asyncpg.Connection, value: dict, seed: str = "seed") -> dict:
    raw = await conn.fetchval(
        "SELECT taleem_select_questions($1,$2,$3,$4::jsonb,$5)",
        *SCOPE,
        json.dumps(value),
        seed,
    )
    return json.loads(raw) if isinstance(raw, str) else raw


@pytest.mark.asyncio
async def test_selector_is_exact_deterministic_and_excludes_unusable_revisions(conn):
    q1, _ = await add_question(conn, chapter="chapter-1", difficulty="easy")
    q2, _ = await add_question(conn, chapter="chapter-1", difficulty="hard")
    q3, _ = await add_question(conn, chapter="chapter-2", difficulty="easy")
    q4, _ = await add_question(conn, chapter="chapter-2", difficulty="hard")
    await add_question(conn, chapter="chapter-1", mode="mcq", marks=1)
    await add_question(conn, chapter="chapter-1", marks=5)
    await add_question(conn, chapter="chapter-1", status="pending")
    await add_question(conn, chapter="chapter-1", superseded=True)
    value = spec(
        [
            section(
                "B",
                select_count=4,
                attempt_count=2,
                chapters={"chapter-1": 2, "chapter-2": 2},
                difficulties={"easy": 2, "hard": 2},
            )
        ]
    )
    first, second = (
        await select(conn, value, "same-seed"),
        await select(conn, value, "same-seed"),
    )
    assert first["satisfiable"] is True
    assert [item["revision_id"] for item in first["selected"]] == [
        item["revision_id"] for item in second["selected"]
    ]
    assert {item["question_id"] for item in first["selected"]} == {q1, q2, q3, q4}
    assert first["total_marks"] == 4


@pytest.mark.asyncio
async def test_selector_rejects_joint_failure_and_never_duplicates(conn):
    await add_question(conn, chapter="chapter-1", difficulty="easy")
    await add_question(conn, chapter="chapter-1", difficulty="easy")
    await add_question(conn, chapter="chapter-2", difficulty="hard")
    await add_question(conn, chapter="chapter-2", difficulty="hard")
    result = await select(
        conn,
        spec(
            [
                section(
                    "B",
                    select_count=2,
                    chapters={"chapter-1": 1, "chapter-2": 1},
                    difficulties={"easy": 2},
                )
            ]
        ),
    )
    assert result["satisfiable"] is False
    assert result["reason"] == "EXACT_ALLOCATION_UNSATISFIED"
    await add_question(conn, chapter="chapter-3", difficulty="medium")
    result = await select(
        conn, spec([section("B1", select_count=1), section("B2", select_count=1)])
    )
    selected_ids = [item["question_id"] for item in result["selected"]]
    assert result["satisfiable"] is True
    assert len(selected_ids) == len(set(selected_ids))


@pytest.mark.asyncio
async def test_selector_seed_changes_the_valid_choice_when_alternatives_exist(conn):
    for index in range(8):
        await add_question(conn, chapter=f"chapter-{index}")
    value = spec([section("B", select_count=2)])
    choices = {
        tuple(
            item["question_id"]
            for item in (await select(conn, value, f"seed-{index}"))["selected"]
        )
        for index in range(12)
    }
    assert len(choices) > 1


@pytest.mark.asyncio
async def test_blueprint_validation_and_activation_are_database_enforced(conn):
    await add_question(conn, chapter="chapter-1")
    valid = spec([section("B", select_count=1)])
    with pytest.raises(asyncpg.RaiseError):
        await conn.execute(
            """INSERT INTO board_paper_blueprints(board_id,class_id,subject_id,name,config,is_active,created_by,updated_by)
                              VALUES($1,$2,$3,'bad',$4::jsonb,FALSE,'test','test')""",
            *SCOPE,
            json.dumps(spec([section("B", select_count=1, attempt_count=2)])),
        )
    await conn.execute(
        """INSERT INTO board_paper_blueprints(board_id,class_id,subject_id,name,config,is_active,created_by,updated_by)
                          VALUES($1,$2,$3,'valid',$4::jsonb,TRUE,'test','test')""",
        *SCOPE,
        json.dumps(valid),
    )
    with pytest.raises(asyncpg.UniqueViolationError):
        await conn.execute(
            """INSERT INTO board_paper_blueprints(board_id,class_id,subject_id,name,config,is_active,created_by,updated_by)
                              VALUES($1,$2,$3,'second',$4::jsonb,TRUE,'test','test')""",
            *SCOPE,
            json.dumps(valid),
        )
    with pytest.raises(asyncpg.RaiseError):
        await conn.execute(
            """INSERT INTO board_paper_blueprints(board_id,class_id,subject_id,name,config,is_active,created_by,updated_by)
               VALUES($1,$2,$3,'unsatisfied',$4::jsonb,TRUE,'test','test')""",
            SCOPE[0],
            "module6-empty-class",
            SCOPE[2],
            json.dumps(valid),
        )
