import pytest

from app.services.runtime_settings import (
    REGISTRY,
    RuntimeSettingError,
    RuntimeSettingsService,
    Scope,
)


def test_registry_is_allowlisted_and_contains_no_secret_or_infrastructure_keys():
    assert "DATABASE_URL" not in REGISTRY
    assert all(
        not any(token in key for token in ("secret", "key", "url", "credential", "jwt"))
        for key in REGISTRY
    )
    with pytest.raises(RuntimeSettingError, match="RUNTIME_SETTING_UNKNOWN_KEY"):
        RuntimeSettingsService._definition("database.url")


def test_types_scopes_bounds_and_hard_ceilings_are_rejected_before_database_work():
    definition = REGISTRY["multiple_ask.max_extracted_questions"]
    with pytest.raises(RuntimeSettingError, match="RUNTIME_SETTING_TYPE_INVALID"):
        RuntimeSettingsService._validate(definition, "20", Scope(kind="global"))
    with pytest.raises(RuntimeSettingError, match="RUNTIME_SETTING_SCOPE_INVALID"):
        RuntimeSettingsService._validate(
            definition, 20, Scope(kind="subject", subject_id="physics")
        )
    with pytest.raises(
        RuntimeSettingError, match="RUNTIME_SETTING_VALUE_OUT_OF_BOUNDS"
    ):
        RuntimeSettingsService._validate(definition, 0, Scope(kind="global"))
    with pytest.raises(
        RuntimeSettingError, match="RUNTIME_SETTING_HARD_CEILING_EXCEEDED"
    ):
        RuntimeSettingsService._validate(definition, 61, Scope(kind="global"))


def test_feature_lifecycle_is_a_small_explicit_enum():
    definition = REGISTRY["feature.multiple_ask"]
    for state in ("disabled", "coming_soon", "enabled"):
        RuntimeSettingsService._validate(definition, state, Scope(kind="global"))
    with pytest.raises(RuntimeSettingError, match="RUNTIME_SETTING_VALUE_INVALID"):
        RuntimeSettingsService._validate(definition, "beta", Scope(kind="global"))
