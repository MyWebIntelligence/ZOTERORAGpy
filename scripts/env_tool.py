#!/usr/bin/env python3
"""Outil du ``.env`` de RAGpy (sprint « configuration unifiée », lot L2).

Commandes (chacune sur une ligne) :

* ``.venv/bin/python scripts/env_tool.py check`` : bilan du ``.env`` (variables
  inconnues, internes, obsolètes, en double, invalides, absentes), **noms
  seulement, jamais de valeur** ;
* ``.venv/bin/python scripts/env_tool.py example`` : régénère ``.env.example``
  depuis le registre (``--check`` : vérifie seulement, code 1 si écart) ;
* ``.venv/bin/python scripts/env_tool.py sync`` : liste les variables obligatoires
  absentes du ``.env`` ; ``--apply`` les y écrit avec la valeur actuelle du code
  (démarrage strict, décision D3), puis range le fichier ;
* ``.venv/bin/python scripts/env_tool.py migrate`` : montre les couples serveur +
  modèle à ajouter pour quitter le mode historique (décision D2 : acheminement
  conservé) ; ``--apply`` les écrit (sauvegarde), puis range le fichier ;
* ``.venv/bin/python scripts/env_tool.py prune`` : liste les variables sans
  effet : obsolètes (jamais lues par le code) et historiques remplacées par un
  couple serveur + modèle qui s'applique ; ``--apply`` les retire du ``.env``
  avec leurs commentaires collés (sauvegarde), des noms donnés en argument
  limitant la liste ;
* ``.venv/bin/python scripts/env_tool.py tidy`` : montre le rangement du ``.env``
  dans l'ordre du registre (noms seulement) ; ``--apply`` l'écrit, après
  sauvegarde dans ``data/config_backups/``, en vérifiant que toutes les valeurs
  restent identiques.

Le registre est ``scripts/rad_settings/registry.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.rad_settings import dotenv_file, registry  # noqa: E402
from scripts.rad_settings.render import (  # noqa: E402
    TIDY_TITLE, check, ignored_historical, migration_plan, render_example, sync_value, tidy_layout,
)

DEFAULT_ENV = REPO / ".env"
DEFAULT_EXAMPLE = REPO / ".env.example"
BACKUP_DIR = REPO / "data" / "config_backups"
LOCK_PATH = REPO / "data" / "locks" / "env.lock"

CATEGORY_TITLES = {
    "error": "Invalides",
    "duplicate": "En double",
    "unknown": "Inconnues du registre",
    "internal": "Internes (à retirer du .env)",
    "obsolete": "Obsolètes (env_tool prune --apply les retire)",
    "planned": "Déclarées avant leur lot (pas encore lues)",
    "warning": "Avertissements",
    "missing": "Obligatoires non déclarées (démarrage refusé : env_tool sync --apply)",
    "ignored": "Historiques sans effet (env_tool prune --apply les retire)",
    "absent": "Facultatives non déclarées (défaut du code appliqué)",
}


def _dotenv_values(path: Path) -> Optional[Dict[str, Optional[str]]]:
    """Valeurs selon python-dotenv (référence), ou ``None`` s'il est absent."""
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - dépendance du dépôt
        return None
    return dict(dotenv_values(path))


def cmd_check(env_path: Path, show_absent: bool) -> int:
    """Affiche le bilan du ``.env`` ; code 1 si invalide, inconnue ou doublon."""
    if not env_path.exists():
        print(f"Fichier absent : {env_path}")
        return 1
    lines = dotenv_file.parse(env_path.read_text(encoding="utf-8"))
    present = dotenv_file.values(lines)
    report = check(present, dotenv_file.duplicates(lines))
    print(f"{env_path.name} : {len(present)} variables déclarées, {len(registry.SETTINGS)} au registre.")
    for category, title in CATEGORY_TITLES.items():
        items = report[category]
        if not items:
            continue
        if category == "absent" and not show_absent:
            print(f"- {title} : {len(items)} (--absent pour la liste)")
            continue
        print(f"- {title} : {len(items)}")
        for name, reason in items:
            print(f"    {name} — {reason}")
    failing = report["error"] or report["unknown"] or report["duplicate"] or report["missing"]
    print("Résultat :", "à corriger" if failing else "conforme")
    return 1 if failing else 0


def cmd_example(example_path: Path, only_check: bool) -> int:
    """Régénère ``.env.example`` (ou vérifie qu'il est à jour)."""
    text = render_example()
    current = example_path.read_text(encoding="utf-8") if example_path.exists() else ""
    if only_check:
        if current == text:
            print(f"{example_path.name} est à jour.")
            return 0
        print(f"{example_path.name} diffère du registre : lancer « python scripts/env_tool.py example ».")
        return 1
    if current == text:
        print(f"{example_path.name} déjà à jour.")
        return 0
    example_path.write_text(text, encoding="utf-8")
    print(f"{example_path.name} régénéré ({len(text.splitlines())} lignes).")
    return 0


def _print_tidy_plan(new_text: str, report: dotenv_file.TidyReport) -> None:
    """Résumé du rangement : en-têtes et noms, jamais de valeur."""
    for line in dotenv_file.parse(new_text):
        if line.kind == dotenv_file.COMMENT and line.is_header:
            print(line.raw)
        elif line.kind == dotenv_file.ASSIGN:
            print(f"    {line.name}")
    print()
    print(f"Variables rangées : {len(report.placed)} ; commentaires collés gardés : {report.kept_comments} ; "
          f"en-têtes et commentaires isolés remplacés : {report.dropped_comments}.")
    for label, names in (("À trier", report.unknown), ("Internes", report.internal), ("Obsolètes", report.obsolete),
                         ("En double (dernière gardée)", report.duplicates)):
        if names:
            print(f"{label} : {', '.join(names)}")
    if report.other_lines:
        print(f"Lignes illisibles ignorées (n° {', '.join(map(str, report.other_lines))}) : à vérifier dans la sauvegarde.")


def cmd_tidy(env_path: Path, apply: bool) -> int:
    """Range le ``.env`` dans l'ordre du registre (aperçu, ou écriture avec ``apply``)."""
    if not env_path.exists():
        print(f"Fichier absent : {env_path}")
        return 1
    original = env_path.read_text(encoding="utf-8")
    lines = dotenv_file.parse(original)
    new_text, report = dotenv_file.tidy(lines, tidy_layout(), registry.classify, TIDY_TITLE)
    before = dotenv_file.values(lines)
    after = dotenv_file.values(dotenv_file.parse(new_text))
    if before != after:
        print("Refus : le rangement changerait des valeurs (analyse interne).")
        return 1
    reference = _dotenv_values(env_path)
    if not apply:
        _print_tidy_plan(new_text, report)
        print("\nAperçu seulement : relancer avec --apply pour écrire (sauvegarde préalable).")
        return 0
    if new_text == original:
        print(f"{env_path.name} déjà rangé.")
        return 0
    backup = dotenv_file.write_in_place(env_path, new_text, backup_dir=BACKUP_DIR, lock_path=LOCK_PATH)
    if reference is not None and _dotenv_values(env_path) != reference:
        if backup is not None:
            dotenv_file.write_in_place(env_path, backup.read_text(encoding="utf-8"), backup_dir=BACKUP_DIR,
                                       lock_path=LOCK_PATH)
        print("Refus : python-dotenv lit des valeurs différentes après rangement ; fichier restauré.")
        return 1
    _print_tidy_plan(new_text, report)
    print(f"\n{env_path.name} rangé ; valeurs identiques (contrôle python-dotenv). Sauvegarde : {backup}")
    return 0


def cmd_sync(env_path: Path, apply: bool) -> int:
    """Écrit dans le ``.env`` les variables obligatoires absentes (valeur actuelle du code)."""
    original = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = dotenv_file.parse(original)
    present = dotenv_file.values(lines)
    missing = [registry.BY_NAME[name] for name in registry.names(registry.ACTIVE)
               if registry.BY_NAME[name].render == "active" and name not in present]
    if not missing:
        print(f"{env_path.name} : aucune variable obligatoire manquante.")
        return 0
    print(f"{len(missing)} variable(s) obligatoire(s) à écrire avec la valeur actuelle du code :")
    for setting in missing:
        note = " (vide : " + setting.default_rule + ")" if setting.default_rule and not sync_value(setting) else ""
        print(f"    {setting.name}{note}")
    if not apply:
        print("\nAperçu seulement : relancer avec --apply pour écrire (sauvegarde préalable).")
        return 0
    added = "\n".join(dotenv_file.format_assignment(s.name, sync_value(s)) for s in missing)
    text = (original.rstrip("\n") + "\n" if original.strip() else "") + added + "\n"
    new_text, _report = dotenv_file.tidy(dotenv_file.parse(text), tidy_layout(), registry.classify, TIDY_TITLE)
    after = dotenv_file.values(dotenv_file.parse(new_text))
    if any(after.get(name) != value for name, value in present.items()):
        print("Refus : l'écriture changerait des valeurs existantes.")
        return 1
    backup = dotenv_file.write_in_place(env_path, new_text, backup_dir=BACKUP_DIR, lock_path=LOCK_PATH)
    print(f"\n{env_path.name} complété et rangé. Sauvegarde : {backup}")
    return 0


def cmd_migrate(env_path: Path, apply: bool) -> int:
    """Ajoute les couples serveur + modèle du mode unifié (valeurs non secrètes affichées)."""
    if not env_path.exists():
        print(f"Fichier absent : {env_path}")
        return 1
    original = env_path.read_text(encoding="utf-8")
    lines = dotenv_file.parse(original)
    present = dotenv_file.values(lines)
    plan = migration_plan(present)
    if not plan:
        print(f"{env_path.name} : rien à migrer (mode unifié déjà déclaré).")
        return 0
    print("Variables ajoutées (adresses et noms de modèles, jamais de clé) :")
    for name, value, reason in plan:
        shown = value if value else "(vide)"
        print(f"    {name}={shown}    # {reason}")
    if not apply:
        print("\nAperçu seulement : relancer avec --apply pour écrire (sauvegarde préalable).")
        return 0
    kept = [line.raw for line in lines if not (line.kind == dotenv_file.ASSIGN and line.name in {n for n, _, _ in plan})]
    text = "\n".join(kept).rstrip("\n") + "\n" + "\n".join(dotenv_file.format_assignment(n, v) for n, v, _ in plan) + "\n"
    new_text, _report = dotenv_file.tidy(dotenv_file.parse(text), tidy_layout(), registry.classify, TIDY_TITLE)
    after = dotenv_file.values(dotenv_file.parse(new_text))
    if any(after.get(name) != value for name, value in present.items() if name not in {n for n, _, _ in plan}):
        print("Refus : la migration changerait des valeurs existantes.")
        return 1
    backup = dotenv_file.write_in_place(env_path, new_text, backup_dir=BACKUP_DIR, lock_path=LOCK_PATH)
    print(f"\n{env_path.name} migré et rangé. Sauvegarde : {backup}")
    return 0


def cmd_prune(env_path: Path, apply: bool, only: Sequence[str]) -> int:
    """Retire du ``.env`` les variables sans effet : obsolètes et historiques remplacées (aperçu, ou ``apply``).

    Args:
        env_path: Fichier ``.env``.
        apply: Écrire (sauvegarde préalable), sinon aperçu.
        only: Noms à retirer (vide : toutes les variables sans effet) ; un nom
            qui n'est pas sans effet est refusé.

    Returns:
        Code de sortie (0 : rien à faire ou fait, 1 : refus).
    """
    if not env_path.exists():
        print(f"Fichier absent : {env_path}")
        return 1
    original = env_path.read_text(encoding="utf-8")
    lines = dotenv_file.parse(original)
    present = dotenv_file.values(lines)
    ignored = {name: f"obsolète, {registry.OBSOLETE_NAMES[name]}" for name in present
               if registry.classify(name) == "obsolete"}
    ignored.update(ignored_historical(present))
    refused = [name for name in only if name not in ignored]
    if refused:
        print("Refus : variable(s) encore lue(s) par le code ou absente(s) du fichier : " + ", ".join(refused))
        return 1
    chosen = [name for name in ignored if not only or name in only]
    if not chosen:
        print(f"{env_path.name} : aucune variable sans effet.")
        return 0
    print(f"{len(chosen)} variable(s) sans effet à retirer :")
    for name in chosen:
        print(f"    {name} — {ignored[name]}")
    if not apply:
        print("\nAperçu seulement : relancer avec --apply pour retirer (sauvegarde préalable).")
        return 0
    removed = set(chosen)
    kept: List[str] = []
    pending: List[str] = []
    for line in lines:
        if line.kind == dotenv_file.COMMENT and not line.is_header and not line.commented_name:
            pending.append(line.raw)
            continue
        if line.kind == dotenv_file.ASSIGN and line.name in removed:
            pending = []  # commentaires collés retirés avec la variable
            continue
        kept.extend(pending)
        pending = []
        kept.append(line.raw)
    kept.extend(pending)
    text = "\n".join(kept).rstrip("\n") + "\n"
    new_text, _report = dotenv_file.tidy(dotenv_file.parse(text), tidy_layout(), registry.classify, TIDY_TITLE)
    after = dotenv_file.values(dotenv_file.parse(new_text))
    expected = {name: value for name, value in present.items() if name not in removed}
    if after != expected:
        print("Refus : le retrait changerait d'autres valeurs.")
        return 1
    backup = dotenv_file.write_in_place(env_path, new_text, backup_dir=BACKUP_DIR, lock_path=LOCK_PATH)
    print(f"\n{env_path.name} : {len(chosen)} variable(s) retirée(s). Sauvegarde : {backup}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Point d'entrée de la ligne de commande."""
    parser = argparse.ArgumentParser(description="Outil du .env de RAGpy (registre : scripts/rad_settings/registry.py)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_check = sub.add_parser("check", help="bilan du .env, noms seulement")
    p_check.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_check.add_argument("--absent", action="store_true", help="lister les variables non déclarées")
    p_example = sub.add_parser("example", help="régénère .env.example depuis le registre")
    p_example.add_argument("--output", type=Path, default=DEFAULT_EXAMPLE)
    p_example.add_argument("--check", action="store_true", help="vérifie seulement (code 1 si écart)")
    p_sync = sub.add_parser("sync", help="écrit les variables obligatoires absentes (valeur actuelle du code)")
    p_sync.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_sync.add_argument("--apply", action="store_true", help="écrit le fichier (sauvegarde préalable)")
    p_migrate = sub.add_parser("migrate", help="ajoute les couples serveur + modèle (sortie du mode historique)")
    p_migrate.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_migrate.add_argument("--apply", action="store_true", help="écrit le fichier (sauvegarde préalable)")
    p_prune = sub.add_parser("prune", help="retire les variables sans effet (obsolètes, historiques remplacées)")
    p_prune.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_prune.add_argument("--apply", action="store_true", help="écrit le fichier (sauvegarde préalable)")
    p_prune.add_argument("names", nargs="*", help="limiter à ces variables")
    p_tidy = sub.add_parser("tidy", help="range le .env dans l'ordre du registre")
    p_tidy.add_argument("--env", type=Path, default=DEFAULT_ENV)
    p_tidy.add_argument("--apply", action="store_true", help="écrit le fichier rangé (sauvegarde préalable)")
    args = parser.parse_args(argv)
    if args.command == "check":
        return cmd_check(args.env, args.absent)
    if args.command == "example":
        return cmd_example(args.output, args.check)
    if args.command == "sync":
        return cmd_sync(args.env, args.apply)
    if args.command == "migrate":
        return cmd_migrate(args.env, args.apply)
    if args.command == "prune":
        return cmd_prune(args.env, args.apply, args.names)
    return cmd_tidy(args.env, args.apply)


if __name__ == "__main__":
    sys.exit(main())
