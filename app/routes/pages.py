"""
Page routes for rendering HTML templates
"""
from typing import Optional
from fastapi import APIRouter, Depends, Request, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import os

from app.database.session import get_db
from app.models.user import User
from app.middleware.auth import get_optional_user

# Templates directory
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(APP_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(tags=["Pages"])


def get_template_context(request: Request, user: Optional[User] = None, **kwargs) -> dict:
    """Creates the base template context"""
    context = {
        "request": request,
        "current_user": user,
        "show_sidebar": kwargs.get("show_sidebar", False),
        "flash_messages": kwargs.get("flash_messages", [])
    }
    context.update(kwargs)
    return context


# --- Public pages ---

@router.get("/login", response_class=HTMLResponse)
async def login_page(
    request: Request,
    verified: Optional[str] = Query(None),
    deleted: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Login page"""
    # Redirect if already logged in
    if current_user:
        return RedirectResponse(url="/", status_code=302)

    context = get_template_context(
        request,
        verified=verified == "true",
        deleted=deleted == "true"
    )
    return templates.TemplateResponse("auth/login.html", context)


@router.get("/register", response_class=HTMLResponse)
async def register_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Registration page"""
    # Redirect if already logged in
    if current_user:
        return RedirectResponse(url="/", status_code=302)

    context = get_template_context(request)
    return templates.TemplateResponse("auth/register.html", context)


@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    """Forgot password page"""
    context = get_template_context(request)
    return templates.TemplateResponse("auth/forgot_password.html", context)


# --- Protected pages ---

@router.get("/profile", response_class=HTMLResponse)
async def profile_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """User profile page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    context = get_template_context(request, current_user)
    return templates.TemplateResponse("user/profile.html", context)


@router.get("/my-projects", response_class=HTMLResponse)
async def my_projects_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """User projects page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    context = get_template_context(request, current_user)
    return templates.TemplateResponse("user/projects.html", context)


# --- Admin pages ---

@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Admin dashboard page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    if not current_user.is_admin:
        return RedirectResponse(url="/", status_code=302)

    context = get_template_context(request, current_user, show_sidebar=True)
    return templates.TemplateResponse("admin/dashboard.html", context)


@router.get("/admin/users", response_class=HTMLResponse)
async def admin_users_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Admin users management page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    if not current_user.is_admin:
        return RedirectResponse(url="/", status_code=302)

    context = get_template_context(request, current_user, show_sidebar=True)
    return templates.TemplateResponse("admin/users.html", context)


@router.get("/admin/projects", response_class=HTMLResponse)
async def admin_projects_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Admin projects management page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    if not current_user.is_admin:
        return RedirectResponse(url="/", status_code=302)

    context = get_template_context(request, current_user, show_sidebar=True)
    # Reuse admin dashboard for now
    return templates.TemplateResponse("admin/dashboard.html", context)


@router.get("/admin/settings", response_class=HTMLResponse)
async def admin_settings_page(
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Admin settings page"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    if not current_user.is_admin:
        return RedirectResponse(url="/", status_code=302)

    context = get_template_context(request, current_user, show_sidebar=True)
    # Reuse admin dashboard for now
    return templates.TemplateResponse("admin/dashboard.html", context)
