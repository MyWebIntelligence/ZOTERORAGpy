"""scripts/rad_env.py — Chargement du ``.env`` gardé pour les scripts lancés en sous-processus.

Les routes lancent ``rad_dataframe.py`` et ``rad_chunk.py`` avec l'environnement
construit par ``app.core.credentials.build_subprocess_env``. Pour un
utilisateur non-admin, cet environnement ne contient plus les clés du ``.env``
serveur ; mais un ``load_dotenv()`` nu, exécuté à l'import du script, les
rechargerait depuis le ``.env`` (il ne remplit que les variables absentes).

``build_subprocess_env`` pose donc ``RAGPY_DOTENV_DENY`` (non-admins seulement,
éventuellement vide si toutes les clés ont été réinjectées) : la liste triée,
séparée par des virgules et sans espace, des noms ``*_API_KEY`` retirés et non
réinjectés. ``load_dotenv_guarded`` charge le ``.env`` comme avant, puis :

* rend aux noms refusés l'état qu'ils avaient avant le chargement (absents
  restent absents) ;
* rend aussi leur valeur d'origine aux variables ``*_API_KEY`` déjà présentes
  avant le chargement (clés personnelles réinjectées), pour qu'un
  ``override=True`` ne les remplace jamais par celles du ``.env`` admin ;
* traite de même les secrets serveur de ``SERVER_SECRET_ENV_VARS`` (clé de
  signature JWT, clé e-mail, mot de passe Flower), que ``build_subprocess_env``
  retire de l'environnement non-admin sans les inscrire dans
  ``RAGPY_DOTENV_DENY`` : absents avant le chargement, ils restent absents.

Conséquences :

* sans ``RAGPY_DOTENV_DENY`` (admin, CLI, tests), le comportement est
  strictement celui de ``dotenv.load_dotenv`` ;
* avec la variable (même vide) et sans ``override=True``, l'environnement
  obtenu est celui de ``dotenv.load_dotenv`` privé des noms refusés et des
  secrets serveur ;
* la configuration non secrète (modèles, URL, réglages) reste relue du ``.env``,
  selon l'``override`` demandé par l'appelant.

Le module vit dans ``scripts/`` : appelé sans chemin explicite, ``find_dotenv()``
remonte depuis ce répertoire, exactement comme depuis ``rad_chunk.py`` ou
``rad_dataframe.py`` ; le fichier résolu est donc le même qu'avant.

Dépendances : stdlib et ``python-dotenv`` seulement ; aucun effet de bord à
l'import.
"""

from __future__ import annotations

import os
from typing import Dict, FrozenSet, Optional

import dotenv

# Nom de la variable posée par ``app.core.credentials.build_subprocess_env``
# (même littéral que ``credentials.DOTENV_DENY_ENV_VAR`` ; un test le vérifie).
DOTENV_DENY_ENV_VAR = "RAGPY_DOTENV_DENY"

# Suffixe des variables secrètes protégées (même convention que la liste de refus).
SECRET_ENV_SUFFIX = "_API_KEY"

# Secrets serveur jamais utiles aux scripts du pipeline : ``build_subprocess_env``
# les retire de l'environnement non-admin (même tuple que
# ``credentials.SERVER_SECRET_ENV_VARS`` ; un test le vérifie). Protégés dès que
# ``RAGPY_DOTENV_DENY`` est présent, pour qu'un ``.env`` ne les réinjecte pas.
SERVER_SECRET_ENV_VARS = ("FLOWER_PASSWORD", "JWT_SECRET_KEY", "JWT_SECRET_KEY_PREVIOUS", "RESEND_API_KEY")


def parse_deny_list(value: Optional[str]) -> FrozenSet[str]:
    """Analyse la valeur de ``RAGPY_DOTENV_DENY`` en ensemble de noms.

    Le format produit par ``build_subprocess_env`` est une liste triée, séparée
    par des virgules, sans espace. Les espaces éventuels et les éléments vides
    sont ignorés, par tolérance.

    Args:
        value: valeur brute de la variable, ou ``None`` si elle est absente.

    Returns:
        Les noms de variables à ne pas recharger depuis le ``.env`` (ensemble
        vide si la variable est absente ou vide).
    """
    if not value:
        return frozenset()
    return frozenset(name.strip() for name in value.split(",") if name.strip())


def load_dotenv_guarded(*args, **kwargs) -> bool:
    """``dotenv.load_dotenv`` qui ne recharge ni ne remplace jamais un secret protégé.

    Sans ``RAGPY_DOTENV_DENY`` dans l'environnement du processus, l'appel est
    strictement identique à ``dotenv.load_dotenv(*args, **kwargs)``.

    Si la variable est présente (même vide, cas d'un non-admin dont toutes les
    clés ont été réinjectées), les noms protégés sont :

    * les noms refusés listés dans ``RAGPY_DOTENV_DENY`` ;
    * les secrets serveur de ``SERVER_SECRET_ENV_VARS`` (retirés par
      ``build_subprocess_env`` sans figurer dans la liste de refus) ;
    * les variables ``*_API_KEY`` déjà présentes avant l'appel (clés
      personnelles réinjectées par ``build_subprocess_env``).

    L'état de chacun (présence et valeur) est noté, ``dotenv.load_dotenv`` est
    appelé avec les mêmes arguments, puis les noms protégés absents avant
    l'appel sont retirés et les autres retrouvent leur valeur d'origine. Ainsi
    ``override=True`` ne peut pas remplacer une clé personnelle par celle du
    ``.env`` admin. Les autres variables du ``.env`` (configuration non
    secrète) sont chargées normalement, ``override`` compris.

    Args:
        *args: arguments positionnels transmis à ``dotenv.load_dotenv``.
        **kwargs: arguments nommés transmis à ``dotenv.load_dotenv``.

    Returns:
        La valeur de retour de ``dotenv.load_dotenv`` (``True`` si au moins une
        variable a été lue dans le fichier).
    """
    raw_deny = os.environ.get(DOTENV_DENY_ENV_VAR)
    if raw_deny is None:
        return dotenv.load_dotenv(*args, **kwargs)

    protected = set(parse_deny_list(raw_deny))
    protected.update(SERVER_SECRET_ENV_VARS)
    protected.update(name for name in os.environ if name.endswith(SECRET_ENV_SUFFIX))
    before: Dict[str, Optional[str]] = {name: os.environ.get(name) for name in protected}
    try:
        return dotenv.load_dotenv(*args, **kwargs)
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
