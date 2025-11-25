"""
Authentication Schemas
======================

This module defines Pydantic schemas for authentication-related operations,
including user registration, login, token management, and password resets.

Key Schemas:
- `UserRegister`: User registration with validation.
- `UserLogin`: User login credentials.
- `TokenResponse`: JWT token response structure.
- `PasswordReset`: Password reset with token validation.
"""
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, field_validator
import re


class UserRegister(BaseModel):
    """Schema pour l'inscription d'un nouvel utilisateur"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    organization: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=255)
    accept_terms: bool = Field(..., description="Doit accepter les conditions d'utilisation")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valide la force du mot de passe"""
        if len(v) < 8:
            raise ValueError("Le mot de passe doit contenir au moins 8 caractères")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Le mot de passe doit contenir au moins une lettre")
        if not re.search(r"\d", v):
            raise ValueError("Le mot de passe doit contenir au moins un chiffre")
        return v

    @field_validator("accept_terms")
    @classmethod
    def must_accept_terms(cls, v: bool) -> bool:
        """Vérifie l'acceptation des conditions"""
        if not v:
            raise ValueError("Vous devez accepter les conditions d'utilisation")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123",
                "first_name": "Jean",
                "last_name": "Dupont",
                "organization": "Université Paris",
                "title": "Chercheur",
                "accept_terms": True
            }
        }


class UserLogin(BaseModel):
    """Schema pour la connexion"""
    email: EmailStr
    password: str
    remember_me: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePass123",
                "remember_me": False
            }
        }


class TokenResponse(BaseModel):
    """Schema pour la réponse d'authentification"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Durée de validité en secondes")

    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class TokenRefresh(BaseModel):
    """Schema pour le rafraîchissement du token"""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Schema pour demander un reset de mot de passe"""
    email: EmailStr

    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com"
            }
        }


class PasswordReset(BaseModel):
    """Schema pour effectuer le reset de mot de passe"""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valide la force du nouveau mot de passe"""
        if len(v) < 8:
            raise ValueError("Le mot de passe doit contenir au moins 8 caractères")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Le mot de passe doit contenir au moins une lettre")
        if not re.search(r"\d", v):
            raise ValueError("Le mot de passe doit contenir au moins un chiffre")
        return v


class PasswordChange(BaseModel):
    """Schema pour changer son mot de passe (utilisateur connecté)"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

    @field_validator("new_password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Valide la force du nouveau mot de passe"""
        if len(v) < 8:
            raise ValueError("Le mot de passe doit contenir au moins 8 caractères")
        if not re.search(r"[A-Za-z]", v):
            raise ValueError("Le mot de passe doit contenir au moins une lettre")
        if not re.search(r"\d", v):
            raise ValueError("Le mot de passe doit contenir au moins un chiffre")
        return v
