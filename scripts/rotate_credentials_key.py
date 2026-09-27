"""Rechiffre les identifiants stockés après un changement de ``JWT_SECRET_KEY`` (audit A09).

Les clés API des utilisateurs (colonne ``users.api_credentials``) sont chiffrées
par Fernet avec une clé dérivée de ``JWT_SECRET_KEY``. Pour changer ce secret
sans rendre ces clés illisibles :

1. définir le nouveau secret dans ``JWT_SECRET_KEY`` et l'ancien dans
   ``JWT_SECRET_KEY_PREVIOUS`` (``.env``), puis redémarrer : l'application lit
   les deux (``MultiFernet``) et chiffre avec le nouveau ;
2. lancer ce script une fois : chaque ligne est déchiffrée (nouveau ou ancien
   secret) et rechiffrée avec le nouveau (``MultiFernet.rotate``) ;
3. vider ``JWT_SECRET_KEY_PREVIOUS`` et redémarrer.

Usage (depuis la racine du dépôt, ou ``docker compose exec ragpy``) ::

    python scripts/rotate_credentials_key.py --dry-run
    python scripts/rotate_credentials_key.py

Aucune valeur de secret n'est jamais affichée : seulement des nombres de lignes.
Code de sortie : 0 succès, 1 lignes illisibles (ni nouveau ni ancien secret),
2 configuration incomplète (``JWT_SECRET_KEY_PREVIOUS`` absent).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def rotate_credentials(db, dry_run: bool = False) -> Dict[str, int]:
    """Rechiffre ``api_credentials`` de chaque utilisateur avec la clé courante.

    Args:
        db: session SQLAlchemy.
        dry_run: ne rien écrire, compter seulement.

    Returns:
        ``{"rotated": n, "unreadable": n, "empty": n}``.
    """
    from cryptography.fernet import InvalidToken

    from app.core.credentials import get_fernet
    from app.models.user import User

    fernet = get_fernet()
    counts = {"rotated": 0, "unreadable": 0, "empty": 0}
    for user in db.query(User).all():
        token = user.api_credentials
        if not token:
            counts["empty"] += 1
            continue
        try:
            rotated = fernet.rotate(token.encode("utf-8")).decode("utf-8")
        except InvalidToken:
            counts["unreadable"] += 1
            continue
        counts["rotated"] += 1
        if not dry_run:
            user.api_credentials = rotated
    if not dry_run:
        db.commit()
    return counts


def main(argv=None) -> int:
    """Point d'entrée : vérifie la configuration, rechiffre, affiche les compteurs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="compter sans écrire")
    args = parser.parse_args(argv)

    from app.config import settings

    if not settings.JWT_SECRET_KEY_PREVIOUS:
        print("JWT_SECRET_KEY_PREVIOUS absent : rien à rechiffrer (définir l'ancien secret d'abord).")
        return 2
    from app.database.session import SessionLocal

    db = SessionLocal()
    try:
        counts = rotate_credentials(db, dry_run=args.dry_run)
    finally:
        db.close()
    mode = "simulation" if args.dry_run else "écrit"
    print(f"Rechiffrement ({mode}) : {counts['rotated']} ligne(s) rechiffrée(s), "
          f"{counts['empty']} sans identifiant, {counts['unreadable']} illisible(s).")
    return 1 if counts["unreadable"] else 0


if __name__ == "__main__":
    sys.exit(main())
