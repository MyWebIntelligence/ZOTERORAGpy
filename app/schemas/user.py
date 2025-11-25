"""
User Management Schemas
=======================

This module defines Pydantic schemas for user management operations, including
user profiles, updates, credentials, and admin statistics.

Key Schemas:
- `UserResponse`: User profile data for API responses.
- `UserUpdate`: Schema for updating user profiles.
- `UserCredentialsResponse`: Masked user credentials for display.
- `UserCredentialsUpdate`: Schema for updating user API credentials.
"""
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Schema de base pour les utilisateurs"""
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    organization: Optional[str] = None
    title: Optional[str] = None


class UserResponse(BaseModel):
    """Schema pour la réponse utilisateur (sans données sensibles)"""
    id: int
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: str
    organization: Optional[str] = None
    title: Optional[str] = None
    roles: List[str] = []
    is_active: bool
    is_verified: bool
    is_admin: bool
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "email": "user@example.com",
                "first_name": "Jean",
                "last_name": "Dupont",
                "full_name": "Jean Dupont",
                "organization": "Université Paris",
                "title": "Chercheur",
                "roles": ["USER", "ADMIN"],
                "is_active": True,
                "is_verified": True,
                "is_admin": True,
                "created_at": "2024-01-15T10:30:00Z",
                "last_login": "2024-01-20T14:00:00Z"
            }
        }


class UserUpdate(BaseModel):
    """Schema pour la mise à jour d'un utilisateur"""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    organization: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=255)

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "Jean",
                "last_name": "Dupont",
                "organization": "Université Lyon",
                "title": "Professeur"
            }
        }


class UserAdminUpdate(BaseModel):
    """Schema pour la mise à jour admin d'un utilisateur"""
    first_name: Optional[str] = Field(None, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    organization: Optional[str] = Field(None, max_length=255)
    title: Optional[str] = Field(None, max_length=255)
    roles: Optional[List[str]] = None
    is_active: Optional[bool] = None
    is_verified: Optional[bool] = None

    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "Jean",
                "last_name": "Dupont",
                "roles": ["USER", "ADMIN"],
                "is_active": True
            }
        }


class UserListResponse(BaseModel):
    """Schema pour la liste paginée des utilisateurs"""
    users: List[UserResponse]
    total: int
    page: int
    per_page: int
    pages: int

    class Config:
        json_schema_extra = {
            "example": {
                "users": [],
                "total": 50,
                "page": 1,
                "per_page": 20,
                "pages": 3
            }
        }


class AdminStats(BaseModel):
    """Schema pour les statistiques du tableau de bord admin"""
    total_users: int
    active_users: int
    admin_users: int
    verified_users: int
    total_projects: int
    recent_logins: int  # Connexions des 7 derniers jours

    class Config:
        json_schema_extra = {
            "example": {
                "total_users": 50,
                "active_users": 45,
                "admin_users": 3,
                "verified_users": 40,
                "total_projects": 25,
                "recent_logins": 30
            }
        }


# --- User Credentials Schemas ---

class CredentialValue(BaseModel):
    """Schema pour une valeur de credential masquee"""
    has_value: bool
    masked: str = ""


class UserCredentialsResponse(BaseModel):
    """Schema pour la reponse des credentials utilisateur (valeurs masquees)"""
    # LLM Providers
    openai_api_key: CredentialValue
    openrouter_api_key: CredentialValue
    openrouter_model: CredentialValue
    # OCR Provider
    mistral_api_key: CredentialValue
    mistral_model: CredentialValue
    mistral_url: CredentialValue
    # Vector Databases
    pinecone_api_key: CredentialValue
    pinecone_env: CredentialValue
    weaviate_api_key: CredentialValue
    weaviate_url: CredentialValue
    qdrant_api_key: CredentialValue
    qdrant_url: CredentialValue
    # Zotero
    zotero_api_key: CredentialValue
    zotero_user_id: CredentialValue
    zotero_group_id: CredentialValue

    class Config:
        json_schema_extra = {
            "example": {
                "openai_api_key": {"has_value": True, "masked": "••••••••abcd"},
                "openrouter_api_key": {"has_value": False, "masked": ""},
                "pinecone_api_key": {"has_value": True, "masked": "••••••••1234"}
            }
        }


class UserCredentialsUpdate(BaseModel):
    """Schema pour la mise a jour des credentials utilisateur"""
    # LLM Providers
    openai_api_key: Optional[str] = Field(None, description="OpenAI API Key")
    openrouter_api_key: Optional[str] = Field(None, description="OpenRouter API Key")
    openrouter_model: Optional[str] = Field(None, description="OpenRouter default model")
    # OCR Provider
    mistral_api_key: Optional[str] = Field(None, description="Mistral API Key")
    mistral_model: Optional[str] = Field(None, description="Mistral model for OCR")
    mistral_url: Optional[str] = Field(None, description="Mistral API base URL")
    # Vector Databases
    pinecone_api_key: Optional[str] = Field(None, description="Pinecone API Key")
    pinecone_env: Optional[str] = Field(None, description="Pinecone environment")
    weaviate_api_key: Optional[str] = Field(None, description="Weaviate API Key")
    weaviate_url: Optional[str] = Field(None, description="Weaviate cluster URL")
    qdrant_api_key: Optional[str] = Field(None, description="Qdrant API Key")
    qdrant_url: Optional[str] = Field(None, description="Qdrant instance URL")
    # Zotero
    zotero_api_key: Optional[str] = Field(None, description="Zotero API Key")
    zotero_user_id: Optional[str] = Field(None, description="Zotero User ID")
    zotero_group_id: Optional[str] = Field(None, description="Zotero Group ID")

    class Config:
        json_schema_extra = {
            "example": {
                "openai_api_key": "sk-xxxxxxxxxxxxxxxxxxxxx",
                "pinecone_api_key": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
            }
        }
