"""Pure usage-policy and Pakistan business-day helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from zoneinfo import ZoneInfo


class AccountTier(StrEnum):
    ANONYMOUS = "anonymous"
    GOOGLE = "google"
    PREMIUM = "premium"


@dataclass(frozen=True)
class UsagePolicy:
    feature: str
    account_tier: AccountTier
    daily_limit: int
    student_visible: bool


@dataclass(frozen=True)
class BusinessWindow:
    business_date: date
    resets_at: datetime
    ttl_seconds: int


@dataclass(frozen=True)
class UsageReservation:
    request_id: str
    used: int
    limit: int
    student_visible: bool
    window: BusinessWindow
    backend: str
    duplicate: bool = False


def pakistan_business_window(now: datetime | None = None) -> BusinessWindow:
    zone = ZoneInfo("Asia/Karachi")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    local = current.astimezone(zone)
    next_day = local.date() + timedelta(days=1)
    reset_local = datetime.combine(next_day, time.min, tzinfo=zone)
    reset_utc = reset_local.astimezone(timezone.utc)
    ttl = max(1, int((reset_utc - current.astimezone(timezone.utc)).total_seconds()))
    return BusinessWindow(
        business_date=local.date(),
        resets_at=reset_utc,
        ttl_seconds=ttl,
    )
