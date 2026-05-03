"""
IP Whitelist / Blacklist API
============================

CRUD endpoints for managing source-IP rules that the analysis engine consults
before running ML inference.

Sync model:
  - PostgreSQL is source of truth.
  - On every mutation we re-publish the active rule set into Redis under
    `ip_rules:whitelist` and `ip_rules:blacklist` (JSON list of {id, cidr}).
  - The engine (`backend/ip_matcher.IPRuleMatcher`) reads those keys on
    startup and refreshes every 30s.

Endpoints (mounted at /api/ip-rules in api.py):

    GET    /api/ip-rules?list_type=whitelist|blacklist   -> view_ip_rules
    POST   /api/ip-rules                                 -> manage_ip_rules
    PATCH  /api/ip-rules/{id}                            -> manage_ip_rules
    DELETE /api/ip-rules/{id}                            -> manage_ip_rules

Frontend handoff:
  - Add a page (suggested: frontend/src/pages/IPRules.jsx) with two tabs
    (whitelist/blacklist), a CRUD table, and an "Add rule" form.
  - Render two new alert types in the alerts list:
      * "Whitelisted" -> grey/benign with a small "rule" badge
      * "Blacklisted" -> red/critical with a "rule" badge
    Both alert payloads carry a top-level `rule_id` referencing IPRule.id.
"""

import ipaddress
import json
from datetime import datetime
from typing import Optional, Literal

import redis.asyncio as redis_async
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from auth import get_current_user, require_permission
from database import get_db
from models import IPRule, User


router = APIRouter()

WHITELIST_KEY = "ip_rules:whitelist"
BLACKLIST_KEY = "ip_rules:blacklist"

ListType = Literal["whitelist", "blacklist"]


# --- Pydantic Schemas ---------------------------------------------------------

class IPRuleCreate(BaseModel):
    cidr: str
    list_type: ListType
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None

    @field_validator("cidr")
    @classmethod
    def _normalize_cidr(cls, v: str) -> str:
        try:
            net = ipaddress.ip_network(v.strip(), strict=False)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Geçersiz CIDR / IP: {v!r} ({e})")
        return str(net)


class IPRuleUpdate(BaseModel):
    is_active: Optional[bool] = None
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None


class IPRuleOut(BaseModel):
    id: int
    cidr: str
    list_type: str
    reason: Optional[str]
    created_by: Optional[str]
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True


# --- Redis publishing ---------------------------------------------------------

async def publish_rules_to_redis(db: AsyncSession, redis_client: redis_async.Redis) -> None:
    """Re-publish the current active rule set into Redis cache keys.

    Called on API startup and after every mutating endpoint so the engine
    sees changes within seconds (not the 30s poll interval).
    """
    now = datetime.utcnow()
    stmt = select(IPRule).where(
        IPRule.is_active.is_(True),
        or_(IPRule.expires_at.is_(None), IPRule.expires_at > now),
    )
    rules = (await db.execute(stmt)).scalars().all()

    buckets: dict[str, list[dict]] = {"whitelist": [], "blacklist": []}
    for r in rules:
        if r.list_type in buckets:
            buckets[r.list_type].append({"id": r.id, "cidr": r.cidr})

    await redis_client.set(WHITELIST_KEY, json.dumps(buckets["whitelist"]))
    await redis_client.set(BLACKLIST_KEY, json.dumps(buckets["blacklist"]))


# --- Endpoints ----------------------------------------------------------------

@router.get("", response_model=list[IPRuleOut])
async def list_rules(
    list_type: Optional[ListType] = None,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(require_permission("view_ip_rules")),
):
    stmt = select(IPRule).order_by(IPRule.id.desc())
    if list_type:
        stmt = stmt.where(IPRule.list_type == list_type)
    rules = (await db.execute(stmt)).scalars().all()
    return rules


@router.post("", response_model=IPRuleOut, status_code=201)
async def create_rule(
    payload: IPRuleCreate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(require_permission("manage_ip_rules")),
):
    rule = IPRule(
        cidr=payload.cidr,
        list_type=payload.list_type,
        reason=payload.reason,
        expires_at=payload.expires_at,
        created_by=user.username,
        is_active=True,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    await publish_rules_to_redis(db, request.app.state.redis)
    return rule


@router.patch("/{rule_id}", response_model=IPRuleOut)
async def update_rule(
    rule_id: int,
    payload: IPRuleUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(require_permission("manage_ip_rules")),
):
    rule = (await db.execute(select(IPRule).where(IPRule.id == rule_id))).scalars().first()
    if not rule:
        raise HTTPException(status_code=404, detail="Kural bulunamadı")

    if payload.is_active is not None:
        rule.is_active = payload.is_active
    if payload.reason is not None:
        rule.reason = payload.reason
    if payload.expires_at is not None:
        rule.expires_at = payload.expires_at

    await db.commit()
    await db.refresh(rule)
    await publish_rules_to_redis(db, request.app.state.redis)
    return rule


@router.delete("/{rule_id}", status_code=204)
async def delete_rule(
    rule_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(require_permission("manage_ip_rules")),
):
    rule = (await db.execute(select(IPRule).where(IPRule.id == rule_id))).scalars().first()
    if not rule:
        raise HTTPException(status_code=404, detail="Kural bulunamadı")
    await db.delete(rule)
    await db.commit()
    await publish_rules_to_redis(db, request.app.state.redis)
    return None
