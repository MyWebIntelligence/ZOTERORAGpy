"""
Email Service using Resend
Handles verification emails and password reset emails
"""
import logging
from pathlib import Path
from typing import Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

try:
    import resend
    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)

# Template directory
TEMPLATES_DIR = Path(__file__).parent.parent / "templates" / "emails"


class EmailService:
    """Service for sending emails via Resend"""

    def __init__(self):
        self._initialized = False
        self._jinja_env: Optional[Environment] = None

    def _initialize(self):
        """Initialize the service (lazy loading)"""
        if self._initialized:
            return

        if not RESEND_AVAILABLE:
            logger.warning("Resend package not installed. Email functionality disabled.")
            self._initialized = True
            return

        if not settings.RESEND_API_KEY:
            logger.warning("RESEND_API_KEY not configured. Email functionality disabled.")
            self._initialized = True
            return

        # Configure Resend
        resend.api_key = settings.RESEND_API_KEY

        # Setup Jinja2 for email templates
        if TEMPLATES_DIR.exists():
            self._jinja_env = Environment(
                loader=FileSystemLoader(str(TEMPLATES_DIR)),
                autoescape=select_autoescape(['html', 'xml'])
            )
        else:
            logger.warning(f"Email templates directory not found: {TEMPLATES_DIR}")

        self._initialized = True
        logger.info("Email service initialized successfully")

    @property
    def is_available(self) -> bool:
        """Check if email service is properly configured"""
        self._initialize()
        return (
            RESEND_AVAILABLE
            and settings.RESEND_API_KEY is not None
            and self._jinja_env is not None
        )

    def _render_template(self, template_name: str, context: dict) -> str:
        """Render an email template with the given context"""
        self._initialize()
        if not self._jinja_env:
            raise RuntimeError("Email templates not configured")

        template = self._jinja_env.get_template(template_name)
        return template.render(**context)

    async def send_verification_email(
        self,
        to_email: str,
        first_name: str,
        verification_token: str
    ) -> bool:
        """
        Send email verification link to user

        Args:
            to_email: User's email address
            first_name: User's first name for personalization
            verification_token: Token for email verification

        Returns:
            True if email sent successfully, False otherwise
        """
        self._initialize()

        if not self.is_available:
            logger.warning("Email service not available, skipping verification email")
            return False

        verification_url = f"{settings.APP_URL}/auth/verify-email/{verification_token}"

        try:
            html_content = self._render_template("verification.html", {
                "first_name": first_name or "Utilisateur",
                "verification_url": verification_url,
                "app_name": settings.APP_NAME,
                "expire_hours": settings.EMAIL_VERIFICATION_EXPIRE_HOURS
            })

            params = {
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": f"[{settings.APP_NAME}] Vérifiez votre adresse email",
                "html": html_content
            }

            response = resend.Emails.send(params)
            logger.info(f"Verification email sent to {to_email}, id: {response.get('id')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send verification email to {to_email}: {e}")
            return False

    async def send_password_reset_email(
        self,
        to_email: str,
        first_name: str,
        reset_token: str,
        initiated_by_admin: bool = False
    ) -> bool:
        """
        Send password reset link to user

        Args:
            to_email: User's email address
            first_name: User's first name for personalization
            reset_token: Token for password reset
            initiated_by_admin: True if reset was initiated by admin

        Returns:
            True if email sent successfully, False otherwise
        """
        self._initialize()

        if not self.is_available:
            logger.warning("Email service not available, skipping password reset email")
            return False

        reset_url = f"{settings.APP_URL}/auth/reset-password/{reset_token}"

        try:
            html_content = self._render_template("password_reset.html", {
                "first_name": first_name or "Utilisateur",
                "reset_url": reset_url,
                "app_name": settings.APP_NAME,
                "expire_hours": settings.PASSWORD_RESET_EXPIRE_HOURS,
                "initiated_by_admin": initiated_by_admin
            })

            subject = f"[{settings.APP_NAME}] Réinitialisation de votre mot de passe"
            if initiated_by_admin:
                subject = f"[{settings.APP_NAME}] Un administrateur a demandé la réinitialisation de votre mot de passe"

            params = {
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": subject,
                "html": html_content
            }

            response = resend.Emails.send(params)
            logger.info(f"Password reset email sent to {to_email}, id: {response.get('id')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send password reset email to {to_email}: {e}")
            return False

    async def send_welcome_email(
        self,
        to_email: str,
        first_name: str
    ) -> bool:
        """
        Send welcome email after successful verification

        Args:
            to_email: User's email address
            first_name: User's first name for personalization

        Returns:
            True if email sent successfully, False otherwise
        """
        self._initialize()

        if not self.is_available:
            logger.warning("Email service not available, skipping welcome email")
            return False

        try:
            # Check if welcome template exists, otherwise skip
            template_path = TEMPLATES_DIR / "welcome.html"
            if not template_path.exists():
                logger.debug("Welcome email template not found, skipping")
                return True

            html_content = self._render_template("welcome.html", {
                "first_name": first_name or "Utilisateur",
                "app_name": settings.APP_NAME,
                "app_url": settings.APP_URL
            })

            params = {
                "from": settings.RESEND_FROM_EMAIL,
                "to": [to_email],
                "subject": f"Bienvenue sur {settings.APP_NAME}!",
                "html": html_content
            }

            response = resend.Emails.send(params)
            logger.info(f"Welcome email sent to {to_email}, id: {response.get('id')}")
            return True

        except Exception as e:
            logger.error(f"Failed to send welcome email to {to_email}: {e}")
            return False


# Singleton instance
email_service = EmailService()
