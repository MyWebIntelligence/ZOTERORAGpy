"""
Token source of the authenticated routes (`app.middleware.auth.get_token_from_request`).

The pages call the API with ``'Bearer ' + localStorage.getItem('access_token')``.
When the browser copy is missing or stale (``Bearer null``) while the
``access_token`` cookie is valid, the header used to win: `/projects` answered
401, the page went to `/login`, which (reading the cookie) sent it back to `/`
— an endless redirect loop (2026-10-02). An undecodable header now yields to
the cookie.

Run with: pytest tests/test_auth_token_source.py -v
"""

from types import SimpleNamespace

from fastapi.security import HTTPAuthorizationCredentials

from app.core.security import create_access_token
from app.middleware.auth import get_token_from_request


def _request(cookie=None):
    """Minimal request double exposing only ``cookies``."""
    return SimpleNamespace(cookies={"access_token": cookie} if cookie else {})


def _bearer(token):
    """``Authorization: Bearer <token>`` as FastAPI's HTTPBearer parses it."""
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_valid_header_wins_over_cookie():
    header, cookie = create_access_token(subject=1), create_access_token(subject=2)
    assert get_token_from_request(_request(cookie), _bearer(header)) == header


def test_bearer_null_yields_to_a_valid_cookie():
    cookie = create_access_token(subject=1)
    assert get_token_from_request(_request(cookie), _bearer("null")) == cookie


def test_stale_header_yields_to_the_cookie():
    cookie = create_access_token(subject=1)
    assert get_token_from_request(_request(cookie), _bearer("eyJ.not.a-valid-jwt")) == cookie


def test_invalid_header_without_cookie_is_still_returned():
    # The caller then answers 401 "Token invalide ou expiré", as before.
    assert get_token_from_request(_request(), _bearer("null")) == "null"


def test_cookie_alone_and_nothing_at_all():
    cookie = create_access_token(subject=1)
    assert get_token_from_request(_request(cookie), None) == cookie
    assert get_token_from_request(_request(), None) is None
