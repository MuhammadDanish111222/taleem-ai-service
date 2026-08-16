"""Strict DTOs for the local runtime-settings control plane."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from app.schemas.ask import StrictModel


class RuntimeSettingScope(StrictModel):
    kind: Literal["global", "subject", "class_subject", "account_tier"]
    subject_id: str | None = Field(default=None, min_length=1, max_length=120)
    class_id: str | None = Field(default=None, min_length=1, max_length=120)
    account_tier: Literal["anonymous", "google", "premium"] | None = None

    @model_validator(mode="after")
    def validate_shape(self):
        if self.kind == "global" and any(
            value is not None
            for value in (self.subject_id, self.class_id, self.account_tier)
        ):
            raise ValueError("RUNTIME_SETTING_SCOPE_INVALID")
        if self.kind == "subject" and (
            not self.subject_id or self.class_id or self.account_tier
        ):
            raise ValueError("RUNTIME_SETTING_SCOPE_INVALID")
        if self.kind == "class_subject" and (
            not self.subject_id or not self.class_id or self.account_tier
        ):
            raise ValueError("RUNTIME_SETTING_SCOPE_INVALID")
        if self.kind == "account_tier" and (
            not self.account_tier or self.subject_id or self.class_id
        ):
            raise ValueError("RUNTIME_SETTING_SCOPE_INVALID")
        return self


class RuntimeSettingMutation(StrictModel):
    key: str = Field(min_length=1, max_length=160, pattern=r"^[a-z0-9][a-z0-9._-]*$")
    scope: RuntimeSettingScope
    value: Any


class RuntimeSettingRead(StrictModel):
    scope: RuntimeSettingScope | None = None
