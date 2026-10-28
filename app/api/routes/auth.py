"""
Authentication API routes.

Endpoints
---------
POST /api/v1/auth/register  – Create a new user account
POST /api/v1/auth/login     – Login (OAuth2 password form) → JWT
POST /api/v1/auth/token     – Alias for /login (OAuth2 spec compat)
GET  /api/v1/auth/me        – Current user info (requires auth)
POST /api/v1/auth/api-key   – Create a programmatic API key
"""
from __future__ import annotations

import re
import secrets
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.security import (
    create_access_token,
    get_current_active_user,
    get_password_hash,
    verify_password,
)
from app.core.config import settings

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# In-memory user store (replace with DB models in production)
# ---------------------------------------------------------------------------
# Keyed by email → {user_id, email, name, hashed_password, is_active, api_keys}
_users: dict[str, dict[str, Any]] = {}

# API keys store: key → {key_id, user_id, name, created_at}
_api_keys: dict[str, dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Minimum 8 characters")
    name: str = Field(..., min_length=1, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one digit.")
        return v


class RegisterResponse(BaseModel):
    user_id: str
    email: str
    access_token: str
    token_type: str = "bearer"


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserInfo(BaseModel):
    user_id: str
    email: str
    name: str
    is_active: bool
    created_at: str


class ApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Label for this API key")


class ApiKeyResponse(BaseModel):
    key_id: str
    api_key: str       # shown only once
    name: str
    created_at: str


# ---------------------------------------------------------------------------
# POST /register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user account",
)
async def register(body: RegisterRequest) -> RegisterResponse:
    email = body.email.lower()

    if email in _users:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    user_id = str(uuid.uuid4())
    _users[email] = {
        "user_id": user_id,
        "email": email,
        "name": body.name,
        "hashed_password": get_password_hash(body.password),
        "is_active": True,
        "created_at": _now_iso(),
    }

    token = create_access_token(
        data={"sub": user_id, "email": email, "is_active": True}
    )

    logger.info("user_registered", user_id=user_id, email=email)
    return RegisterResponse(user_id=user_id, email=email, access_token=token)


# ---------------------------------------------------------------------------
# POST /login  (JSON body convenience endpoint)
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post(
    "/login",
    response_model=LoginResponse,
    summary="Login with email and password",
)
async def login_json(body: LoginRequest) -> LoginResponse:
    return _authenticate(body.email, body.password)


# ---------------------------------------------------------------------------
# POST /token  (OAuth2 form – used by Swagger UI)
# ---------------------------------------------------------------------------

@router.post(
    "/token",
    response_model=LoginResponse,
    summary="OAuth2 token endpoint (Swagger UI)",
)
async def login_form(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> LoginResponse:
    return _authenticate(form_data.username, form_data.password)


def _authenticate(email: str, password: str) -> LoginResponse:
    """Shared authentication logic."""
    user = _users.get(email.lower())
    if not user or not verify_password(password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )

    token = create_access_token(
        data={"sub": user["user_id"], "email": user["email"], "is_active": True}
    )
    logger.info("user_logged_in", user_id=user["user_id"])
    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_EXPIRE_MINUTES * 60,
    )


# ---------------------------------------------------------------------------
# GET /me
# ---------------------------------------------------------------------------

@router.get(
    "/me",
    response_model=UserInfo,
    summary="Get current user info",
)
async def get_me(
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
) -> UserInfo:
    email = current_user.get("email", "")
    user = _users.get(email.lower())
    if not user:
        raise HTTPException(status_code=404, detail="User record not found.")

    return UserInfo(
        user_id=user["user_id"],
        email=user["email"],
        name=user["name"],
        is_active=user["is_active"],
        created_at=user["created_at"],
    )


# ---------------------------------------------------------------------------
# POST /api-key
# ---------------------------------------------------------------------------

@router.post(
    "/api-key",
    response_model=ApiKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a programmatic API key",
)
async def create_api_key(
    body: ApiKeyRequest,
    current_user: Annotated[dict[str, Any], Depends(get_current_active_user)],
) -> ApiKeyResponse:
    user_id = current_user.get("sub", "")
    key_id = str(uuid.uuid4())
    raw_key = f"mra_{secrets.token_urlsafe(32)}"

    _api_keys[raw_key] = {
        "key_id": key_id,
        "user_id": user_id,
        "name": body.name,
        "created_at": _now_iso(),
    }

    logger.info("api_key_created", user_id=user_id, key_id=key_id, name=body.name)
    return ApiKeyResponse(
        key_id=key_id,
        api_key=raw_key,  # shown only once
        name=body.name,
        created_at=_api_keys[raw_key]["created_at"],
    )
