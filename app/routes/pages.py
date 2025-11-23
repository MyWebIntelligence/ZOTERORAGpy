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
from app.models.project import Project
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


# --- Project pages ---

@router.get("/project/{project_id}", response_class=HTMLResponse)
async def project_detail_page(
    project_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Project detail page - redirects to the main pipeline interface"""
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    # Get the project
    project = db.query(Project).filter(Project.id == project_id).first()

    if not project:
        # Project not found - redirect to projects list
        return RedirectResponse(url="/my-projects", status_code=302)

    # Check access
    user_role = project.get_user_role(current_user.id)
    if not user_role and not current_user.is_admin:
        # No access - redirect to projects list
        return RedirectResponse(url="/my-projects", status_code=302)

    # Get project owner
    owner = db.query(User).filter(User.id == project.owner_id).first()

    context = get_template_context(
        request,
        current_user,
        project=project,
        owner=owner,
        user_role=user_role or "admin"
    )
    return templates.TemplateResponse("user/project_detail.html", context)


@router.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(
    request: Request,
    project: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Pipeline RAG page - the main processing interface.
    If project_id is provided, links uploads to that project.
    """
    if not current_user:
        return RedirectResponse(url="/login", status_code=302)

    project_data = None
    if project:
        project_data = db.query(Project).filter(Project.id == project).first()
        if project_data:
            # Check access
            user_role = project_data.get_user_role(current_user.id)
            if not user_role and not current_user.is_admin:
                return RedirectResponse(url="/my-projects", status_code=302)

    context = get_template_context(
        request,
        current_user,
        project=project_data,
        project_id=project
    )
    return templates.TemplateResponse("index.html", context)
