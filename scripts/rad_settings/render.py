"""Rendus tirés du registre : ``.env.example``, plan de rangement, contrôle (lot L2).

Bibliothèque standard seulement ; utilisé par ``scripts/env_tool.py`` et par
les tests de synchronisation.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Mapping, Tuple

from . import registry as reg

EXAMPLE_PREAMBLE = """\
# =============================================================================
# RAGpy — modèle de configuration
# =============================================================================
# Copier en .env à la racine du dépôt : cp .env.example .env
#
# Fichier GÉNÉRÉ par « python scripts/env_tool.py example » à partir du registre
# scripts/rad_settings/registry.py : ne pas l'éditer à la main (modifier le
# registre, puis régénérer). Même ordre et mêmes blocs que le .env rangé par
# « python scripts/env_tool.py tidy --apply ».
#
# Qui lit le .env :
#   - l'application web (app/config.py, load_dotenv) ;
#   - les scripts du pipeline (scripts/rad_env.load_dotenv_guarded) ;
#   - Docker Compose : env_file des 4 services (ragpy, celery_worker,
#     celery_beat, flower) ET interpolation des ${…} de docker-compose.yml.
#   Exception : scripts/rad_vectordb.py en ligne de commande ne charge pas le
#   .env (exporter les variables dans le shell).
#
# Conventions :
#   - La valeur écrite est le défaut du code, sauf si le commentaire indique
#     « défaut : … ». Supprimer une ligne redonne le défaut du code ; une ligne
#     laissée vide ne le fait que si le commentaire le précise.
#   - « # NOM=valeur » : variable facultative, non déclarée ; décommenter la
#     ligne pour l'utiliser.
#   - Serveurs d'API (bloc 1) : OpenRouter et Mistral préremplis, les autres en
#     commentaire. Adresses vérifiées, documentation et modèles réputés :
#     README.md, section « Serveurs d'API et modèles ». Les adresses des
#     nouveaux serveurs et les couples serveur + modèle par service (bloc 2)
#     sont lus à partir du lot 4 du sprint « configuration unifiée ».
#   - Clés API : le .env sert de repli aux seuls administrateurs. Un utilisateur
#     non-admin n'y a jamais accès et saisit ses clés dans
#     Paramètres > Mes Identifiants (chiffrées en base avec JWT_SECRET_KEY).
#   - Docker : le code est intégré à l'image (rebuild après mise à jour du
#     code) ; le .env est relu à la recréation des conteneurs
#     (docker compose up -d). INSTALL_LOCAL_OCR, lu à la construction, exige
#     docker compose up -d --build.
#
# Minimum pour démarrer : décommenter OPENAI_API_KEY (embeddings OpenAI par
# défaut) et, en production, le bloc 9 (RAGPY_ENV, JWT_SECRET_KEY, CORS).
"""

TIDY_TITLE = (
    "# =============================================================================",
    "# RAGpy — configuration locale (secrets : ne jamais versionner)",
    "# Rangée par « python scripts/env_tool.py tidy --apply » dans l'ordre du registre",
    "# scripts/rad_settings/registry.py. Descriptions de chaque variable : .env.example.",
    "# =============================================================================",
)


def _comment(text: str) -> List[str]:
    """Lignes de commentaire ``# …`` pour un texte (lignes vides : ``#``)."""
    return [("# " + line).rstrip() for line in text.splitlines()]


def _assignment(setting: reg.Setting) -> str:
    """Ligne d'exemple d'une variable (``NOM=valeur`` ou ``# NOM=valeur``)."""
    line = f"{setting.name}={setting.example_value}"
    return "# " + line if setting.render == "commented" else line


def render_example() -> str:
    """Contenu complet de ``.env.example``, tiré du registre."""
    out: List[str] = [EXAMPLE_PREAMBLE.rstrip("\n")]
    for block in reg.LAYOUT:
        block_out: List[str] = []
        for sub in block.subblocks:
            shown = [reg.BY_NAME[s.name] for s in sub.settings if s.render != "hidden"]
            if not shown:
                continue
            block_out.append("")
            block_out.append(f"# --- {sub.key} {sub.title} ---")
            block_out.extend(_comment(sub.doc) if sub.doc else [])
            for setting in shown:
                block_out.extend(_comment(setting.doc) if setting.doc else [])
                block_out.append(_assignment(setting))
        if not block_out:
            continue
        out.append("")
        out.append(f"# ===== {block.key}. {block.title} =====")
        out.extend(_comment(block.doc) if block.doc else [])
        out.extend(block_out)
    return "\n".join(out) + "\n"


def tidy_layout() -> List[Tuple[str, str, List[Tuple[str, str, List[str]]]]]:
    """Plan de rangement : ``[(bloc, titre, [(sous-bloc, titre, [noms])])]``."""
    return [
        (block.key, block.title, [(sub.key, sub.title, [s.name for s in sub.settings]) for sub in block.subblocks])
        for block in reg.LAYOUT
    ]


_BOOL_LITERALS = {
    "01": ("0", "1"),
    "truefalse": ("true", "false"),
    "TRUEFALSE": ("TRUE", "FALSE"),
}
_URL_SCHEMES = ("http://", "https://", "redis://", "rediss://", "sqlite://", "postgresql://", "postgres://")


def validate(setting: reg.Setting, value: str) -> Tuple[str, str]:
    """Contrôle de forme d'une valeur, sans jamais la renvoyer.

    Args:
        setting: Variable du registre.
        value: Valeur lue dans le ``.env``.

    Returns:
        ``("ok" | "warning" | "error", raison)`` ; une valeur vide est toujours
        admise ici (le défaut du code s'applique jusqu'au lot L3).
    """
    if value == "":
        return "ok", ""
    kind = setting.kind
    if kind == reg.BOOL:
        literals = _BOOL_LITERALS.get(setting.bool_style, ())
        if value in literals:
            return "ok", ""
        if value.lower() in ("0", "1", "true", "false", "yes", "no", "on", "off"):
            return "warning", f"booléen non canonique (attendu : {' ou '.join(literals)})"
        return "error", "booléen illisible"
    if kind == reg.INT:
        return ("ok", "") if re.fullmatch(r"-?\d+", value) else ("error", "entier attendu")
    if kind == reg.FLOAT:
        try:
            float(value)
        except ValueError:
            return "error", "nombre attendu"
        return "ok", ""
    if kind == reg.ENUM:
        return ("ok", "") if value in setting.choices else ("error", f"valeur hors choix ({', '.join(setting.choices)})")
    if kind == reg.URL:
        return ("ok", "") if value.startswith(_URL_SCHEMES) else ("error", "adresse attendue (http, https, redis…)")
    return "ok", ""


def sync_value(setting: reg.Setting) -> str:
    """Valeur écrite par ``env_tool sync`` : le défaut actuel du code.

    Un défaut calculé est évalué sur ce poste (``DEFAULT_MAX_WORKERS`` = nombre
    de CPU − 1, au moins 1) ; les autres défauts calculés acceptent une valeur
    vide (« vide = règle du code », voir ``default_rule``).
    """
    if setting.name == "DEFAULT_MAX_WORKERS":
        return str(max(1, (os.cpu_count() or 2) - 1))
    return setting.default


ALWAYS_REPLACED_PREFIXES = ("LLM_DEFAULT_", "LLM_RECODE_", "LLM_NOTES_", "LLM_BOOK_", "LLM_CITATIONS_")
"""Couples qui s'appliquent toujours en mode unifié (repli sur LLM_DEFAULT_SERVER + LLM_DEFAULT_MODEL)."""


def ignored_historical(present: Mapping[str, str]) -> List[Tuple[str, str]]:
    """Variables historiques présentes dans le ``.env`` mais sans effet.

    En mode unifié (``LLM_DEFAULT_SERVER`` déclaré), une variable remplacée
    (``replaced_by``) n'est plus lue quand son couple s'applique : toujours pour
    le recodage, les notes, les fiches, les citations et le défaut ; pour les
    autres services (OCR, réponses sourcées, embeddings, rerank, transcription),
    seulement quand le modèle du couple est déclaré (sinon la règle historique
    vaut encore). Une variable historique laissée à la valeur par défaut du
    code est aussi sans effet : la retirer ne change rien.

    Args:
        present: Valeurs du ``.env``.

    Returns:
        ``[(nom, raison)]`` dans l'ordre du registre ; vide en mode historique.
    """
    if not (present.get("LLM_DEFAULT_SERVER") or "").strip():
        return []
    out: List[Tuple[str, str]] = []
    for setting in reg.SETTINGS:
        if not setting.replaced_by or setting.name not in present:
            continue
        always = any(name.startswith(ALWAYS_REPLACED_PREFIXES) for name in setting.replaced_by)
        triggers = [name for name in setting.replaced_by if name.endswith("_MODEL")] or list(setting.replaced_by)
        if always or any((present.get(name) or "").strip() for name in triggers):
            out.append((setting.name, "sans effet, remplacée par " + " + ".join(setting.replaced_by)))
        elif (present.get(setting.name) or "").strip() == setting.default and setting.default:
            out.append((setting.name, "sans effet, valeur par défaut du code (remplacée par "
                        + " + ".join(setting.replaced_by) + ")"))
    return out


def check(present: Mapping[str, str], duplicated: List[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Bilan d'un ``.env`` : noms classés et raisons, **jamais de valeur**.

    Args:
        present: Valeurs du ``.env`` (nom → valeur).
        duplicated: Noms affectés plusieurs fois.

    Returns:
        Table catégorie → ``[(nom, raison)]`` avec les catégories ``error``,
        ``warning``, ``unknown``, ``internal``, ``obsolete``, ``duplicate``,
        ``planned`` (déclarée avant son lot), ``missing`` (obligatoire non
        déclarée : refus du démarrage strict), ``ignored`` (historique sans effet
        en mode unifié, ``ignored_historical``) et ``absent`` (facultative non
        déclarée : défaut du code appliqué).
    """
    report: Dict[str, List[Tuple[str, str]]] = {
        key: [] for key in ("error", "warning", "unknown", "internal", "obsolete", "duplicate", "planned", "missing",
                            "ignored", "absent")
    }
    report["ignored"] = ignored_historical(present)
    for name in duplicated:
        report["duplicate"].append((name, "affectée plusieurs fois (la dernière l'emporte)"))
    for name, value in present.items():
        category = reg.classify(name)
        if category == "internal":
            report["internal"].append((name, "interne : à retirer du .env"))
            continue
        if category == "obsolete":
            report["obsolete"].append((name, reg.OBSOLETE_NAMES[name]))
            continue
        if category == "unknown":
            report["unknown"].append((name, "inconnue du registre"))
            continue
        setting = reg.BY_NAME[name]
        if setting.status == reg.PLANNED:
            report["planned"].append((name, "pas encore lue par le code"))
        level, reason = validate(setting, value or "")
        if level != "ok":
            report[level].append((name, reason))
    for setting in reg.SETTINGS:
        if setting.status == reg.ACTIVE and setting.name not in present:
            if setting.render == "active":
                report["missing"].append((setting.name, "obligatoire : env_tool sync --apply l'écrit"))
            else:
                report["absent"].append((setting.name, "facultative, non déclarée : défaut du code appliqué"))
    return report


OFFICIAL_SERVER_URLS = {
    "OPENROUTER_API_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENAI_API_BASE_URL": "https://api.openai.com/v1",
}
"""Adresses écrites par ``migrate`` pour les serveurs que le mode historique utilisait."""


def migration_plan(values: Mapping[str, str]) -> List[Tuple[str, str, str]]:
    """Variables à ajouter pour passer du mode historique à la règle serveur + modèle (D2).

    Seules les variables absentes sont proposées ; l'acheminement d'aujourd'hui
    est conservé :

    * adresses d'OpenRouter et d'OpenAI (utilisées implicitement avant) ;
    * ``LLM_DEFAULT_SERVER`` / ``LLM_DEFAULT_MODEL`` séparés depuis
      ``OPENROUTER_DEFAULT_MODEL`` (vide : OpenAI ``gpt-4o-mini``, défaut du code) ;
    * recodage : gabarit d'Amar, ``openai/gpt-4o-mini`` sur le serveur par
      défaut (``RECODE_MODEL`` séparé si le mode durci est actif).

    Args:
        values: Valeurs du ``.env`` actuel.

    Returns:
        ``[(nom, valeur, raison)]`` dans l'ordre du registre ; jamais de secret.
    """
    from .chat import ALBERT_DEFAULT_BASE_URL
    from .models import split_legacy

    def present(name: str) -> bool:
        return bool((values.get(name) or "").strip())

    plan: Dict[str, Tuple[str, str]] = {}
    for name, url in OFFICIAL_SERVER_URLS.items():
        if not present(name):
            plan[name] = (url, "adresse officielle, utilisée implicitement en mode historique")
    openai_url = values.get("OPENAI_API_BASE_URL") or OFFICIAL_SERVER_URLS["OPENAI_API_BASE_URL"]
    openrouter_url = values.get("OPENROUTER_API_BASE_URL") or OFFICIAL_SERVER_URLS["OPENROUTER_API_BASE_URL"]
    albert_url = values.get("ALBERT_BASE_URL") or ALBERT_DEFAULT_BASE_URL
    if present("LLM_DEFAULT_SERVER"):
        _plan_ocr(values, plan, albert_url, present)
        return [(n, v, r) for n in reg.names() if n in plan for v, r in [plan[n]]]
    legacy_default = (values.get("OPENROUTER_DEFAULT_MODEL") or "").strip() or "gpt-4o-mini"
    server, model = split_legacy(legacy_default, openai_url=openai_url, openrouter_url=openrouter_url,
                                 albert_url=albert_url)
    plan["LLM_DEFAULT_SERVER"] = (server, f"séparé depuis OPENROUTER_DEFAULT_MODEL ({legacy_default})")
    plan["LLM_DEFAULT_MODEL"] = (model, "idem")
    harden = (values.get("RECODE_HARDEN_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")
    if harden and present("RECODE_MODEL"):
        r_server, r_model = split_legacy(values["RECODE_MODEL"], openai_url=openai_url, openrouter_url=openrouter_url,
                                         albert_url=albert_url)
        plan["LLM_RECODE_SERVER"] = (r_server, "mode durci : séparé depuis RECODE_MODEL")
        plan["LLM_RECODE_MODEL"] = (r_model, "idem")
    elif not present("LLM_RECODE_MODEL"):
        plan["LLM_RECODE_SERVER"] = ("", "vide : serveur par défaut")
        plan["LLM_RECODE_MODEL"] = ("openai/gpt-4o-mini", "gabarit d'Amar (gpt-4o-mini par OpenRouter)")
    _plan_ocr(values, plan, albert_url, present)
    return [(n, v, r) for n in reg.names() if n in plan for v, r in [plan[n]]]


def _plan_ocr(values: Mapping[str, str], plan: Dict[str, Tuple[str, str]], albert_url: str, present) -> None:
    """Règle OCR de ``migration_plan`` (lot L5) : le premier maillon actif de la chaîne
    historique devient le seul moteur déclaré (Albert, sinon Mistral, sinon Docling,
    sinon PyMuPDF)."""
    if not present("OCR_MODEL"):
        # Lot L5 : le maillon qui ouvre la chaîne historique devient le seul moteur.
        on = ("1", "true", "yes", "on")
        albert_ocr = ((values.get("ALBERT_ENABLED") or "").strip().lower() in on
                      and (values.get("OCR_ENABLE_ALBERT") or "").strip().lower() in on
                      and present("ALBERT_API_KEY"))
        if albert_ocr:
            plan["OCR_SERVER"] = (albert_url, "OCR Albert en tête de la chaîne historique")
            plan["OCR_MODEL"] = ((values.get("ALBERT_OCR_CHAT_MODEL") or "lightonocr-2-1b").strip(),
                                 "LightOnOCR (le compte n'a pas accès à /v1/ocr)")
        elif present("MISTRAL_API_KEY"):
            from .chat import mistral_base_url

            plan["OCR_SERVER"] = (mistral_base_url(values.get("MISTRAL_API_BASE_URL") or "https://api.mistral.ai"),
                                  "OCR Mistral en tête de la chaîne historique")
            plan["OCR_MODEL"] = ((values.get("MISTRAL_OCR_MODEL") or "mistral-ocr-latest").strip(),
                                 "depuis MISTRAL_OCR_MODEL")
        elif (values.get("INSTALL_LOCAL_OCR") or "").strip().lower() in on:
            plan["OCR_SERVER"] = ("local", "OCR local Docling installé")
            plan["OCR_MODEL"] = ("docling", "idem")
        else:
            plan["OCR_SERVER"] = ("local", "aucune clé d'OCR : couche texte native")
            plan["OCR_MODEL"] = ("pymupdf", "idem (PDF scannés : choisir un moteur d'OCR)")
