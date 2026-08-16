"""Allowlisted, typed runtime settings over the existing owner tables.

This module intentionally does not expose environment configuration, credentials,
or the grounded-evidence rule.  PostgreSQL is the invalidation authority; callers
may safely use defaults when the optional runtime table is not available yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import asyncpg

ScopeKind = Literal["global", "subject", "class_subject", "account_tier"]
Owner = Literal["system_settings", "usage_policies", "ask_source_policies"]


class RuntimeSettingError(ValueError):
    """Stable, safe rejection code returned by the admin endpoint."""


@dataclass(frozen=True)
class Scope:
    kind: ScopeKind
    subject_id: str | None = None
    class_id: str | None = None
    account_tier: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Scope":
        return cls(
            kind=value["kind"],
            subject_id=value.get("subject_id"),
            class_id=value.get("class_id"),
            account_tier=value.get("account_tier"),
        )

    def token(self) -> str:
        return "|".join(
            (
                self.kind,
                self.subject_id or "",
                self.class_id or "",
                self.account_tier or "",
            )
        )

    def as_dict(self) -> dict[str, str | None]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "class_id": self.class_id,
            "account_tier": self.account_tier,
        }


@dataclass(frozen=True)
class SettingDefinition:
    key: str
    owner: Owner
    value_type: Literal["integer", "boolean", "enum", "number"]
    default: Any
    scopes: tuple[ScopeKind, ...]
    description: str
    cache_namespace: str
    minimum: int | float | None = None
    maximum: int | float | None = None
    hard_ceiling: int | float | None = None
    enum_values: tuple[str, ...] = ()
    mutable: bool = True
    owner_column: str | None = None


# The registry is intentionally small and explicit. Adding a credential-like
# name here is rejected by construction and must never be used as an escape hatch
# for environment configuration.
REGISTRY: dict[str, SettingDefinition] = {
    **{
        f"usage.single_question.daily_limit.{tier}": SettingDefinition(
            key=f"usage.single_question.daily_limit.{tier}",
            owner="usage_policies",
            value_type="integer",
            default=default,
            scopes=("account_tier",),
            description=f"Single Ask daily limit for {tier} accounts.",
            cache_namespace="usage_policy",
            minimum=0,
            maximum=10_000,
            hard_ceiling=10_000,
            owner_column="daily_limit",
        )
        for tier, default in (("anonymous", 5), ("google", 5), ("premium", 10_000))
    },
    **{
        f"usage.multiple_question_batch.daily_limit.{tier}": SettingDefinition(
            key=f"usage.multiple_question_batch.daily_limit.{tier}",
            owner="usage_policies",
            value_type="integer",
            default=default,
            scopes=("account_tier",),
            description=f"Multiple Ask daily batch limit for {tier} accounts.",
            cache_namespace="usage_policy",
            minimum=0,
            maximum=1_000,
            hard_ceiling=1_000,
            owner_column="daily_limit",
        )
        for tier, default in (("anonymous", 0), ("google", 1), ("premium", 1_000))
    },
    "ask.general_fallback": SettingDefinition(
        "ask.general_fallback",
        "ask_source_policies",
        "boolean",
        False,
        ("global", "subject", "class_subject"),
        "Allow clearly labelled General AI fallback after weak or absent scoped evidence.",
        "ask_source_policy",
        owner_column="allow_general",
    ),
    "ask.semantic_reuse_enabled": SettingDefinition(
        "ask.semantic_reuse_enabled",
        "ask_source_policies",
        "boolean",
        False,
        ("global", "subject", "class_subject"),
        "Allow approved semantic answer reuse within the selected scope.",
        "ask_source_policy",
        owner_column="semantic_reuse_enabled",
    ),
    "ask.semantic_distance_threshold": SettingDefinition(
        "ask.semantic_distance_threshold",
        "ask_source_policies",
        "number",
        0.18,
        ("global", "subject", "class_subject"),
        "Maximum semantic distance for approved-answer reuse when reuse is enabled.",
        "ask_source_policy",
        minimum=0,
        maximum=2,
        hard_ceiling=2,
        owner_column="semantic_distance_threshold",
    ),
    "retrieval.dense_candidate_count": SettingDefinition(
        "retrieval.dense_candidate_count",
        "system_settings",
        "integer",
        10,
        ("global",),
        "Dense retrieval candidate count.",
        "runtime_setting",
        minimum=1,
        maximum=50,
        hard_ceiling=50,
    ),
    "retrieval.expected_question_candidate_count": SettingDefinition(
        "retrieval.expected_question_candidate_count",
        "system_settings",
        "integer",
        10,
        ("global",),
        "Expected-question retrieval candidate count.",
        "runtime_setting",
        minimum=1,
        maximum=50,
        hard_ceiling=50,
    ),
    "retrieval.lexical_candidate_count": SettingDefinition(
        "retrieval.lexical_candidate_count",
        "system_settings",
        "integer",
        10,
        ("global",),
        "Lexical retrieval candidate count.",
        "runtime_setting",
        minimum=1,
        maximum=50,
        hard_ceiling=50,
    ),
    "multiple_ask.answer_batch_size": SettingDefinition(
        "multiple_ask.answer_batch_size",
        "system_settings",
        "integer",
        5,
        ("global",),
        "Maximum Multiple Ask answers processed per worker batch.",
        "runtime_setting",
        minimum=1,
        maximum=10,
        hard_ceiling=10,
    ),
    "multiple_ask.max_extracted_questions": SettingDefinition(
        "multiple_ask.max_extracted_questions",
        "system_settings",
        "integer",
        60,
        ("global",),
        "Maximum questions extracted from one Multiple Ask submission.",
        "runtime_setting",
        minimum=1,
        maximum=60,
        hard_ceiling=60,
    ),
    "multiple_ask.ocr_timeout_seconds": SettingDefinition(
        "multiple_ask.ocr_timeout_seconds",
        "system_settings",
        "integer",
        20,
        ("global",),
        "Maximum OCR provider wait time for one operation.",
        "runtime_setting",
        minimum=1,
        maximum=30,
        hard_ceiling=30,
    ),
    "test_generation.max_duration_minutes": SettingDefinition(
        "test_generation.max_duration_minutes",
        "system_settings",
        "integer",
        600,
        ("global",),
        "Maximum duration allowed for a generated test paper.",
        "runtime_setting",
        minimum=1,
        maximum=600,
        hard_ceiling=600,
    ),
    "feature.multiple_ask": SettingDefinition(
        "feature.multiple_ask",
        "system_settings",
        "enum",
        "disabled",
        ("global",),
        "Student lifecycle state for Multiple Ask.",
        "feature_state",
        enum_values=("disabled", "coming_soon", "enabled"),
    ),
    "feature.test_generation": SettingDefinition(
        "feature.test_generation",
        "system_settings",
        "enum",
        "enabled",
        ("global",),
        "Student lifecycle state for test-paper generation.",
        "feature_state",
        enum_values=("disabled", "coming_soon", "enabled"),
    ),
}


def _feature_for_usage(key: str) -> str:
    return (
        "single_question"
        if key.startswith("usage.single_question.")
        else "multiple_question_batch"
    )


def _tier_for_usage(key: str) -> str:
    """The tier embedded in a usage key is the sole tier authority."""
    return key.rsplit(".", 1)[-1]


class RuntimeSettingsService:
    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    @staticmethod
    def metadata() -> list[dict[str, Any]]:
        return [
            {
                "key": item.key,
                "owner": item.owner,
                "value_type": item.value_type,
                "default": item.default,
                "minimum": item.minimum,
                "maximum": item.maximum,
                "hard_ceiling": item.hard_ceiling,
                "allowed_values": list(item.enum_values),
                "scopes": list(item.scopes),
                "mutable": item.mutable,
                "cache_namespace": item.cache_namespace,
                "description": item.description,
            }
            for item in REGISTRY.values()
        ]

    @staticmethod
    def _definition(key: str) -> SettingDefinition:
        definition = REGISTRY.get(key)
        if definition is None:
            raise RuntimeSettingError("RUNTIME_SETTING_UNKNOWN_KEY")
        return definition

    @staticmethod
    def _validate(definition: SettingDefinition, value: Any, scope: Scope) -> None:
        if scope.kind not in definition.scopes:
            raise RuntimeSettingError("RUNTIME_SETTING_SCOPE_INVALID")
        if (
            definition.owner == "usage_policies"
            and scope.account_tier != _tier_for_usage(definition.key)
        ):
            raise RuntimeSettingError("RUNTIME_SETTING_TIER_MISMATCH")
        if definition.value_type == "boolean" and type(value) is not bool:
            raise RuntimeSettingError("RUNTIME_SETTING_TYPE_INVALID")
        if definition.value_type == "integer" and (
            type(value) is not int or isinstance(value, bool)
        ):
            raise RuntimeSettingError("RUNTIME_SETTING_TYPE_INVALID")
        if definition.value_type == "number" and (
            type(value) not in (int, float) or isinstance(value, bool)
        ):
            raise RuntimeSettingError("RUNTIME_SETTING_TYPE_INVALID")
        if definition.value_type == "enum" and (
            type(value) is not str or value not in definition.enum_values
        ):
            raise RuntimeSettingError("RUNTIME_SETTING_VALUE_INVALID")
        if definition.minimum is not None and value < definition.minimum:
            raise RuntimeSettingError("RUNTIME_SETTING_VALUE_OUT_OF_BOUNDS")
        if definition.hard_ceiling is not None and value > definition.hard_ceiling:
            raise RuntimeSettingError("RUNTIME_SETTING_HARD_CEILING_EXCEEDED")
        if definition.maximum is not None and value > definition.maximum:
            raise RuntimeSettingError("RUNTIME_SETTING_VALUE_OUT_OF_BOUNDS")

    @staticmethod
    def _system_key(definition: SettingDefinition, scope: Scope) -> str:
        return f"runtime:{definition.key}:{scope.token()}"

    async def get(self, key: str, scope: Scope) -> Any:
        definition = self._definition(key)
        if scope.kind not in definition.scopes:
            raise RuntimeSettingError("RUNTIME_SETTING_SCOPE_INVALID")
        try:
            if definition.owner == "usage_policies":
                row = await self._conn.fetchrow(
                    "SELECT daily_limit FROM usage_policies WHERE feature=$1 AND account_tier=$2",
                    _feature_for_usage(key),
                    scope.account_tier,
                )
                return row["daily_limit"] if row else definition.default
            if definition.owner == "ask_source_policies":
                row = await self._conn.fetchrow(
                    """SELECT allow_general, semantic_reuse_enabled, semantic_distance_threshold
                       FROM ask_source_policies WHERE class_id IS NOT DISTINCT FROM $1 AND subject_id IS NOT DISTINCT FROM $2""",
                    scope.class_id,
                    scope.subject_id,
                )
                if row is None:
                    return definition.default
                value = row[definition.owner_column or ""]
                return definition.default if value is None else value
            raw = await self._conn.fetchval(
                "SELECT value FROM system_settings WHERE key=$1",
                self._system_key(definition, scope),
            )
            if raw is None:
                return definition.default
            return json.loads(raw) if isinstance(raw, str) else raw
        # Unit-level fakes and startup-degraded workers may not provide a live
        # asyncpg connection. Runtime values are optional by design, so their
        # absence must be indistinguishable from an unavailable settings table.
        except (
            AttributeError,
            asyncpg.UndefinedTableError,
            asyncpg.UndefinedColumnError,
        ):
            return definition.default

    async def list_effective(self) -> list[dict[str, Any]]:
        result = []
        for definition in REGISTRY.values():
            for kind in definition.scopes:
                if kind != "global":
                    continue
                scope = Scope(kind="global")
                result.append(
                    {
                        "key": definition.key,
                        "scope": scope.as_dict(),
                        "value": await self.get(definition.key, scope),
                    }
                )
        return result

    async def set(
        self, *, key: str, scope: Scope, value: Any, actor_id: str, request_id: str
    ) -> dict[str, Any]:
        definition = self._definition(key)
        self._validate(definition, value, scope)
        async with self._conn.transaction():
            before = await self.get(key, scope)
            if definition.owner == "usage_policies":
                await self._conn.execute(
                    "UPDATE usage_policies SET daily_limit=$1, updated_by=$2, updated_at=NOW() WHERE feature=$3 AND account_tier=$4",
                    value,
                    actor_id,
                    _feature_for_usage(key),
                    scope.account_tier,
                )
                cache_key = f"{_feature_for_usage(key)}:{scope.account_tier}"
            elif definition.owner == "ask_source_policies":
                await self._conn.execute(
                    """INSERT INTO ask_source_policies(class_id,subject_id,allow_general,semantic_reuse_enabled,semantic_distance_threshold,updated_by)
                       VALUES($1,$2,FALSE,FALSE,NULL,$3)
                       ON CONFLICT (COALESCE(class_id, ''), COALESCE(subject_id, '')) DO NOTHING""",
                    scope.class_id,
                    scope.subject_id,
                    actor_id,
                )
                if definition.owner_column == "semantic_distance_threshold":
                    await self._conn.execute(
                        "UPDATE ask_source_policies SET semantic_distance_threshold=$1, semantic_reuse_enabled=TRUE, updated_by=$2, updated_at=NOW() WHERE class_id IS NOT DISTINCT FROM $3 AND subject_id IS NOT DISTINCT FROM $4",
                        float(value),
                        actor_id,
                        scope.class_id,
                        scope.subject_id,
                    )
                elif definition.owner_column == "semantic_reuse_enabled":
                    await self._conn.execute(
                        "UPDATE ask_source_policies SET semantic_reuse_enabled=$1, semantic_distance_threshold=CASE WHEN $1 THEN COALESCE(semantic_distance_threshold, 0.18) ELSE NULL END, updated_by=$2, updated_at=NOW() WHERE class_id IS NOT DISTINCT FROM $3 AND subject_id IS NOT DISTINCT FROM $4",
                        value,
                        actor_id,
                        scope.class_id,
                        scope.subject_id,
                    )
                else:
                    await self._conn.execute(
                        "UPDATE ask_source_policies SET allow_general=$1, updated_by=$2, updated_at=NOW() WHERE class_id IS NOT DISTINCT FROM $3 AND subject_id IS NOT DISTINCT FROM $4",
                        value,
                        actor_id,
                        scope.class_id,
                        scope.subject_id,
                    )
                cache_key = scope.token()
            else:
                await self._conn.execute(
                    """INSERT INTO system_settings(key,value,description,updated_by,revision,updated_at)
                       VALUES($1,$2::jsonb,$3,$4,1,NOW())
                       ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, description=EXCLUDED.description,
                         updated_by=EXCLUDED.updated_by, revision=system_settings.revision+1, updated_at=NOW()""",
                    self._system_key(definition, scope),
                    json.dumps(value),
                    definition.description,
                    actor_id,
                )
                cache_key = scope.token()
            generation = await self._conn.fetchval(
                """INSERT INTO cache_generations(namespace,cache_key,generation) VALUES($1,$2,2)
                   ON CONFLICT(namespace,cache_key) DO UPDATE SET generation=cache_generations.generation+1,updated_at=NOW()
                   RETURNING generation""",
                definition.cache_namespace,
                cache_key,
            )
            revision = (
                await self._conn.fetchval(
                    "SELECT revision FROM system_settings WHERE key=$1",
                    self._system_key(definition, scope),
                )
                if definition.owner == "system_settings"
                else int(generation)
            )
            record = {
                "setting_key": key,
                "scope": scope.as_dict(),
                "previous_value": before,
                "new_value": value,
                "revision": int(revision),
                "request_id": request_id,
                "cache_generation": int(generation),
            }
            await self._conn.execute(
                """INSERT INTO runtime_setting_audits(setting_key,scope,previous_value,new_value,actor_id,revision,request_id,cache_namespace,cache_generation)
                VALUES($1,$2::jsonb,$3::jsonb,$4::jsonb,$5,$6,$7,$8,$9)""",
                key,
                json.dumps(scope.as_dict()),
                json.dumps(before),
                json.dumps(value),
                actor_id,
                int(revision),
                request_id,
                definition.cache_namespace,
                int(generation),
            )
            await self._conn.execute(
                """INSERT INTO admin_audit_logs(actor_id,action,target_type,target_id,before_value,after_value)
                VALUES($1,'runtime_setting.updated','runtime_setting',$2,$3::jsonb,$4::jsonb)""",
                actor_id,
                f"{key}:{scope.token()}",
                json.dumps({"value": before}),
                json.dumps(record),
            )
        return record
