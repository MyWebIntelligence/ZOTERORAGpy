"""
Pydantic schemas for user management
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
