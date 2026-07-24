"""JSONL Chunk Validator and Parser for Taleem AI Service."""

import json
import hashlib
import logging
from typing import Dict, Any, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)

VALID_CONTENT_TYPES: Set[str] = {
    "explanation",
    "definition",
    "worked_example",
    "formula",
    "summary",
    "exercise",
}

VALID_LANGUAGES: Set[str] = {
    "en",
    "ur",
    "roman_ur",
    "mixed",
}


def count_tokens(text: str) -> int:
    """Computes token count for text using word count fallback."""
    if not text:
        return 0
    return len(text.split())


def compute_content_hash(text: str) -> str:
    """Computes SHA-256 hex digest of chunk text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def check_firestore_hierarchy(
    firestore_db: Any,
    board_id: str,
    class_id: str,
    subject_id: str,
    chapter_id: str,
    cache: Dict[Tuple[str, str, str, str], bool],
    allow_mock_validation_for_tests: bool = False,
) -> bool:
    """Checks full ancestor chain in Firestore: board -> class -> subject -> chapter.

    Verifies document existence and active == True across all 4 levels.
    Uses in-memory batch caching to prevent redundant Firestore network reads.
    """
    key = (board_id, class_id, subject_id, chapter_id)
    if key in cache:
        return cache[key]

    if firestore_db is None:
        if allow_mock_validation_for_tests:
            cache[key] = True
            return True
        raise RuntimeError("Firestore DB instance is required for catalogue hierarchy verification.")

    try:
        # 1. Board check
        board_ref = firestore_db.collection("boards").document(board_id)
        board_doc = await board_ref.get() if hasattr(board_ref.get, "__await__") else board_ref.get()
        if not board_doc.exists:
            cache[key] = False
            return False
        board_data = board_doc.to_dict() or {}
        if board_data.get("active") is False:
            cache[key] = False
            return False

        # 2. Class check
        class_ref = board_ref.collection("classes").document(class_id)
        class_doc = await class_ref.get() if hasattr(class_ref.get, "__await__") else class_ref.get()
        if not class_doc.exists:
            cache[key] = False
            return False
        class_data = class_doc.to_dict() or {}
        if class_data.get("active") is False:
            cache[key] = False
            return False

        # 3. Subject check
        subject_ref = class_ref.collection("subjects").document(subject_id)
        subject_doc = await subject_ref.get() if hasattr(subject_ref.get, "__await__") else subject_ref.get()
        if not subject_doc.exists:
            cache[key] = False
            return False
        subject_data = subject_doc.to_dict() or {}
        if subject_data.get("active") is False:
            cache[key] = False
            return False

        # 4. Chapter check
        chapter_ref = subject_ref.collection("chapters").document(chapter_id)
        chapter_doc = await chapter_ref.get() if hasattr(chapter_ref.get, "__await__") else chapter_ref.get()
        if not chapter_doc.exists:
            cache[key] = False
            return False
        chapter_data = chapter_doc.to_dict() or {}
        if chapter_data.get("active") is False:
            cache[key] = False
            return False

        cache[key] = True
        return True
    except Exception as err:
        logger.error(f"Error during Firestore hierarchy check for scope {board_id}/{class_id}/{subject_id}/{chapter_id}: {err}")
        cache[key] = False
        return False


async def validate_and_parse_jsonl(
    raw_content: str,
    firestore_db: Optional[Any] = None,
    allow_mock_validation_for_tests: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Validates line-by-line JSONL input for admin chunk ingestion.

    Returns:
        (valid_chunks, errors):
        - valid_chunks: List of validated chunk dicts if errors is empty.
        - errors: List of sanitized per-row error dicts [{"row": int, "field": str, "reason": str}].
    """
    errors: List[Dict[str, Any]] = []
    parsed_rows: List[Tuple[int, Dict[str, Any]]] = []

    lines = raw_content.splitlines()
    seen_chunk_orders: Set[Tuple[str, str, str, str, int]] = set()
    hierarchy_cache: Dict[Tuple[str, str, str, str], bool] = {}

    for row_idx, line in enumerate(lines, start=1):
        line_str = line.strip()
        if not line_str:
            continue

        try:
            row_data = json.loads(line_str)
        except Exception:
            errors.append({"row": row_idx, "field": "raw_line", "reason": "invalid_json"})
            continue

        if not isinstance(row_data, dict):
            errors.append({"row": row_idx, "field": "raw_line", "reason": "row_must_be_json_object"})
            continue

        # Validate required fields
        board_id = row_data.get("board_id")
        if not board_id or not isinstance(board_id, str):
            errors.append({"row": row_idx, "field": "board_id", "reason": "missing_or_invalid_string"})

        class_id = row_data.get("class_id")
        if not class_id or not isinstance(class_id, str):
            errors.append({"row": row_idx, "field": "class_id", "reason": "missing_or_invalid_string"})

        subject_id = row_data.get("subject_id")
        if not subject_id or not isinstance(subject_id, str):
            errors.append({"row": row_idx, "field": "subject_id", "reason": "missing_or_invalid_string"})

        chapter_id = row_data.get("chapter_id")
        if not chapter_id or not isinstance(chapter_id, str):
            errors.append({"row": row_idx, "field": "chapter_id", "reason": "missing_or_invalid_string"})

        topic_no = row_data.get("topic_no")
        if topic_no is None:
            errors.append({"row": row_idx, "field": "topic_no", "reason": "missing_topic_no"})
        else:
            topic_no = str(topic_no).strip()
            if not topic_no:
                errors.append({"row": row_idx, "field": "topic_no", "reason": "empty_topic_no"})

        topic_title = row_data.get("topic_title")
        if not topic_title or not isinstance(topic_title, str):
            errors.append({"row": row_idx, "field": "topic_title", "reason": "missing_or_invalid_string"})

        chunk_order = row_data.get("chunk_order")
        if chunk_order is None or not isinstance(chunk_order, int) or isinstance(chunk_order, bool) or chunk_order < 0:
            errors.append({"row": row_idx, "field": "chunk_order", "reason": "must_be_non_negative_integer"})

        content_type = row_data.get("content_type")
        if not content_type or not isinstance(content_type, str) or content_type not in VALID_CONTENT_TYPES:
            errors.append({"row": row_idx, "field": "content_type", "reason": "invalid_content_type_enum"})

        chunk_text = row_data.get("chunk_text")
        if not chunk_text or not isinstance(chunk_text, str) or not chunk_text.strip():
            errors.append({"row": row_idx, "field": "chunk_text", "reason": "missing_or_empty_chunk_text"})

        expected_questions = row_data.get("expected_questions")
        if expected_questions is not None:
            if not isinstance(expected_questions, list) or not all(isinstance(q, str) for q in expected_questions):
                errors.append({"row": row_idx, "field": "expected_questions", "reason": "must_be_list_of_strings"})

        page_range = row_data.get("page_range")
        if page_range is not None:
            if (
                not isinstance(page_range, list)
                or len(page_range) != 2
                or isinstance(page_range[0], bool)
                or isinstance(page_range[1], bool)
                or not isinstance(page_range[0], int)
                or not isinstance(page_range[1], int)
                or page_range[0] < 1
                or page_range[1] < page_range[0]
            ):
                errors.append({
                    "row": row_idx,
                    "field": "page_range",
                    "reason": "page_range_must_be_null_or_array_of_two_integers_start_end"
                })

        language = row_data.get("language", "en")
        if not isinstance(language, str) or language not in VALID_LANGUAGES:
            errors.append({"row": row_idx, "field": "language", "reason": "invalid_language"})

        # Check duplicate chunk_order per scope
        if (
            isinstance(board_id, str)
            and isinstance(class_id, str)
            and isinstance(subject_id, str)
            and isinstance(chapter_id, str)
            and isinstance(chunk_order, int)
            and not isinstance(chunk_order, bool)
            and chunk_order >= 0
        ):
            scope_key = (board_id, class_id, subject_id, chapter_id, chunk_order)
            if scope_key in seen_chunk_orders:
                errors.append({"row": row_idx, "field": "chunk_order", "reason": "duplicate_chunk_order_in_batch"})
            else:
                seen_chunk_orders.add(scope_key)

        # Hierarchy check against Firestore (if field basic types valid)
        if (
            isinstance(board_id, str)
            and isinstance(class_id, str)
            and isinstance(subject_id, str)
            and isinstance(chapter_id, str)
        ):
            hierarchy_valid = await check_firestore_hierarchy(
                firestore_db,
                board_id,
                class_id,
                subject_id,
                chapter_id,
                hierarchy_cache,
                allow_mock_validation_for_tests=allow_mock_validation_for_tests,
            )
            if not hierarchy_valid:
                errors.append({"row": row_idx, "field": "chapter_id", "reason": "unknown_or_inactive_catalogue_hierarchy"})

        parsed_rows.append((row_idx, row_data))

    if errors:
        return [], errors

    valid_chunks: List[Dict[str, Any]] = []
    for row_idx, row in parsed_rows:
        text = row["chunk_text"].strip()
        page_start = row["page_range"][0] if row.get("page_range") else None
        page_end = row["page_range"][1] if row.get("page_range") else None

        chunk = {
            "board_id": row["board_id"],
            "class_id": row["class_id"],
            "subject_id": row["subject_id"],
            "chapter_id": row["chapter_id"],
            "topic_no": str(row["topic_no"]).strip(),
            "topic_title": row["topic_title"].strip(),
            "chunk_order": row["chunk_order"],
            "content_type": row["content_type"],
            "chunk_text": text,
            "expected_questions": row.get("expected_questions") or [],
            "page_start": page_start,
            "page_end": page_end,
            "language": row.get("language", "en"),
            "content_hash": compute_content_hash(text),
            "token_count": count_tokens(text),
            "metadata": row.get("metadata") or {},
        }
        valid_chunks.append(chunk)

    return valid_chunks, []
