"""Outils communs des fiches d'évaluation humaine Albert (RAGpy).

Format repris du protocole validé de ``moisson_shs`` (``scripts/eval/`` et
``docs/EVALUATION.md``) :

- fiche **CSV UTF-8 avec BOM** (séparateur « ; », ouvrable dans un tableur) ;
- page **HTML autonome hors ligne** (aucun script, style ni police externes),
  notation au clavier, jugements gardés dans le navigateur (``localStorage``,
  toujours dans un ``try``), bouton « Exporter le CSV » qui produit le fichier
  à rendre (``<fiche>_rempli.csv``) ;
- échelles explicites et affichées sur la page ;
- ordre aléatoire à **graine fixe**, second jugement partiel (≈ 20 %, autre
  ordre, autres identifiants) pour mesurer l'accord (kappa de Cohen pondéré
  quadratique, implémenté ici sans dépendance).

Aucun chemin absolu n'est écrit dans les fiches : ``scrub_paths`` ramène tout
chemin local à son nom de fichier.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import os
import random
import re
import shutil
import tempfile
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[3]
"""Racine du dépôt RAGpy."""

EVAL_ROOT = REPO / "data" / "albert_eval"
"""Racine des campagnes (``data/`` est exclu du dépôt git)."""

DEFAULT_SEED = 20261002
DEFAULT_SECOND_RATIO = 0.2
DELIMITER = ";"
BACKUPS = "sauvegardes"

OCR_SCALE: Tuple[Tuple[int, str, str], ...] = (
    (0, "illisible/faux", "texte absent, inutilisable ou sans rapport avec la page"),
    (1, "nombreuses erreurs", "le sens est atteint : mots, chiffres ou lignes manquants ou faux"),
    (2, "erreurs mineures", "quelques coquilles, le sens est intact"),
    (3, "fidèle", "transcription exacte (la mise en forme ne compte pas)"),
)
"""Échelle de fidélité d'une transcription OCR (valeur, libellé, définition)."""

AUDIO_SCALE: Tuple[Tuple[int, str, str], ...] = (
    (0, "inaudible/faux", "extrait inaudible, ou transcription absente, fausse ou sans rapport"),
    (1, "nombreuses erreurs", "le sens est atteint : mots manquants, faux ou inventés"),
    (2, "erreurs mineures", "quelques mots ou noms propres erronés, le sens est intact"),
    (3, "fidèle", "transcription exacte (ponctuation et hésitations ne comptent pas)"),
)
"""Échelle de fidélité d'une transcription audio (valeur, libellé, définition)."""

PREFERENCES: Tuple[Tuple[str, str, str], ...] = (
    ("A", "A", "la version A est plus fidèle et plus lisible"),
    ("B", "B", "la version B est plus fidèle et plus lisible"),
    ("égal", "égal", "pas de différence qui compte"),
)
"""Choix de la comparaison A/B (valeur, libellé, définition)."""

_ABS_PATH_RE = re.compile(
    r"(?<![\w.~-])(?:[A-Za-z]:\\|/)(?:Users|home|tmp|private|var|opt|Volumes|mnt|srv|root|data|app)"
    r"[\\/][^\s\"'<>|]+"
)
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


class EvalError(Exception):
    """Erreur d'usage ou de données, signalée en français sans trace Python."""


# ---------------------------------------------------------------------------
# Chemins et fichiers
# ---------------------------------------------------------------------------
def default_out_dir(today: Optional[date] = None) -> Path:
    """Dossier de campagne par défaut : ``data/albert_eval/<AAAA-MM-JJ>``.

    Args:
        today: date de la campagne (défaut : aujourd'hui).

    Returns:
        Le chemin du dossier (non créé).
    """
    return EVAL_ROOT / (today or date.today()).isoformat()


def display_path(path: Path) -> str:
    """Chemin affichable dans la console : relatif au dépôt si possible.

    Args:
        path: chemin à afficher.

    Returns:
        Le chemin relatif à la racine du dépôt, sinon le chemin tel quel.
    """
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _prefix_variants(prefixes: Iterable[Any]) -> List[str]:
    """Préfixes de dossiers à retirer (forme donnée et forme résolue), du plus long au plus court."""
    out = set()
    for prefix in prefixes:
        if not prefix:
            continue
        for form in (str(prefix), os.path.realpath(str(prefix))):
            form = form.rstrip("\\/")
            if len(form) > 1:
                out.add(form)
    return sorted(out, key=len, reverse=True)


def scrub_paths(text: Any, prefixes: Iterable[Any] = ()) -> str:
    """Ramène tout chemin local absolu d'un texte à son nom de fichier.

    Les dossiers connus (``prefixes``, dépôt, dossier personnel) sont traités
    d'abord, du plus long au plus court, ce qui couvre les dossiers dont le
    nom contient des espaces ; tout autre chemin absolu est ensuite reconnu
    par sa racine (``/Users``, ``/home``, ``/tmp``, ``C:\\``…).

    Args:
        text: texte à nettoyer (``None`` donne ``""``).
        prefixes: dossiers supplémentaires (par exemple ``--pdf-dir``).

    Returns:
        Le texte sans chemin absolu (``/Users/x/doc.pdf`` devient ``doc.pdf``).
    """
    value = "" if text is None else str(text)

    def _basename(match: "re.Match[str]") -> str:
        """Nom de fichier du chemin trouvé (vide si le chemin est le dossier lui-même)."""
        rest = match.groupdict().get("rest")
        target = match.group(0) if rest is None else rest
        tail = [p for p in re.split(r"[\\/]", target.rstrip("\\/")) if p]
        return tail[-1] if tail else ""

    for prefix in _prefix_variants([*prefixes, REPO, Path.home()]):
        value = re.sub(re.escape(prefix) + r"(?P<rest>[\\/][^\s\"'<>|]*)?(?![^\s\\/\"'<>|])", _basename, value)
    return _ABS_PATH_RE.sub(_basename, value)


def write_text(path: Path, text: str, *, bom: bool = False) -> Path:
    """Écrit un fichier texte UTF-8 de façon atomique (fichier temporaire puis renommage).

    Args:
        path: destination.
        text: contenu.
        bom: préfixer d'un BOM UTF-8 (fiches CSV pour tableur).

    Returns:
        Le chemin écrit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig" if bom else "utf-8", newline="") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def csv_cell(value: Any, *, text: bool = False) -> str:
    """Valeur de cellule CSV : sauts de ligne aplatis, formule de tableur neutralisée.

    Args:
        value: valeur brute.
        text: colonne de texte libre (une espace est ajoutée devant ``=``,
            ``+``, ``-`` ou ``@`` initial, pour qu'un tableur n'y voie pas une
            formule).

    Returns:
        La chaîne écrite dans la cellule.
    """
    cell = "" if value is None else str(value)
    cell = cell.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    if text and cell.startswith(_FORMULA_START):
        cell = " " + cell
    return cell


def rows_to_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str],
                text_columns: Iterable[str] = ()) -> str:
    """Texte CSV (séparateur « ; », guillemets si besoin, fins de ligne CRLF).

    Args:
        rows: lignes (clés = colonnes ; les autres clés sont ignorées).
        columns: colonnes, dans l'ordre.
        text_columns: colonnes de texte libre (voir ``csv_cell``).

    Returns:
        Le texte CSV, sans BOM.
    """
    textual = set(text_columns)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=DELIMITER, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    writer.writerow(list(columns))
    for row in rows:
        writer.writerow([csv_cell(row.get(c, ""), text=c in textual) for c in columns])
    return buf.getvalue()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str],
              text_columns: Iterable[str] = ()) -> Path:
    """Écrit une fiche CSV UTF-8 avec BOM (voir ``rows_to_csv``).

    Args:
        path: destination.
        rows: lignes.
        columns: colonnes.
        text_columns: colonnes de texte libre.

    Returns:
        Le chemin écrit.
    """
    return write_text(path, rows_to_csv(rows, columns, text_columns), bom=True)


def _raise_field_limit() -> None:
    """Relève la taille maximale d'un champ CSV (textes OCR de plusieurs Mo)."""
    limit = 2 ** 31 - 1
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def read_csv_rows(path: Path) -> Tuple[List[Tuple[int, Dict[str, str]]], List[str]]:
    """Lit un CSV rempli (tableur ou export HTML) de façon tolérante.

    BOM toléré, séparateur détecté sur l'en-tête (« ; », « , » ou
    tabulation), en-têtes ramenés en minuscules sans espaces autour. Un
    fichier non UTF-8 (format « CSV » simple d'un tableur) est relu en
    Windows-1252 avec un avertissement. Une ligne ayant plus de champs que
    l'en-tête est écartée et signalée (colonnes possiblement décalées).

    Args:
        path: fichier CSV.

    Returns:
        ``(lignes, avertissements)`` ; chaque ligne est ``(numéro de ligne du
        fichier, valeurs)``, l'en-tête étant la ligne 1.

    Raises:
        EvalError: fichier absent ou illisible.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EvalError(f"fichier illisible : {path.name} ({exc.strerror or exc})") from None
    warnings: List[str] = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        line = raw[:exc.start].count(b"\n") + 1
        text = raw.decode("cp1252", errors="replace")
        warnings.append(f"{path.name} : fichier non encodé en UTF-8 (ligne {line}), relu en Windows-1252 ; "
                        "réenregistrer au format « CSV UTF-8 » de préférence")
    _raise_field_limit()
    first = text.splitlines()[0] if text else ""
    delim = max((DELIMITER, ",", "\t"), key=first.count) if first else DELIMITER
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delim)
    rows: List[Tuple[int, Dict[str, str]]] = []
    for row in reader:
        if None in row:
            warnings.append(f"{path.name} ligne {reader.line_num} : plus de champs que l'en-tête "
                            f"(séparateur « {delim} » saisi sans guillemets ?) ; ligne écartée")
            continue
        rows.append((reader.line_num,
                     {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}))
    return rows, warnings


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Écrit un fichier JSON Lines (un objet par ligne, UTF-8).

    Args:
        path: destination.
        records: objets sérialisables.

    Returns:
        Le chemin écrit.
    """
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    return write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Lit un fichier JSON Lines (lignes vides ignorées).

    Args:
        path: fichier à lire.

    Returns:
        Les objets, dans l'ordre.

    Raises:
        EvalError: fichier absent ou ligne JSON invalide (numéro de ligne cité).
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EvalError(f"fichier illisible : {path.name} ({exc.strerror or exc})") from None
    out: List[Dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(f"{path.name} ligne {number} : JSON invalide ({exc.msg})") from None
        if not isinstance(value, dict):
            raise EvalError(f"{path.name} ligne {number} : objet JSON attendu")
        out.append(value)
    return out


def has_judgments(path: Path, fields: Sequence[str] = ("note", "preference", "commentaire")) -> bool:
    """Vrai si une fiche CSV existante contient au moins une saisie.

    Args:
        path: fiche CSV.
        fields: colonnes de saisie.

    Returns:
        ``True`` si une cellule de saisie est remplie.
    """
    if not Path(path).exists():
        return False
    try:
        rows, _warnings = read_csv_rows(path)
    except EvalError:
        return True
    return any(row.get(f, "") for _line, row in rows for f in fields)


def backup_file(path: Path) -> Path:
    """Copie horodatée d'une fiche avant réécriture (``sauvegardes/<nom>_<horodatage>``).

    Args:
        path: fichier à copier.

    Returns:
        Le chemin de la copie.
    """
    path = Path(path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest_dir = path.parent / BACKUPS
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{path.stem}_{stamp}{path.suffix}"
    n = 2
    while dest.exists():
        dest = dest_dir / f"{path.stem}_{stamp}-{n}{path.suffix}"
        n += 1
    shutil.copy2(path, dest)
    return dest


def guard_overwrite(paths: Sequence[Path], force: bool) -> List[Path]:
    """Refuse d'écraser une fiche déjà remplie, sauf ``force`` (copie gardée).

    Args:
        paths: fiches CSV qui vont être réécrites.
        force: écraser malgré des saisies (une copie est faite avant).

    Returns:
        Les copies de sauvegarde faites.

    Raises:
        EvalError: fiche remplie et ``force`` absent (rien n'est écrit).
    """
    filled = [p for p in paths if has_judgments(p)]
    if filled and not force:
        names = ", ".join(p.name for p in filled)
        raise EvalError(f"{names} contient déjà des jugements : rien n'a été écrit. Exporter d'abord les "
                        "jugements, ou relancer avec --force (une copie sera gardée dans sauvegardes/).")
    return [backup_file(p) for p in filled]


# ---------------------------------------------------------------------------
# Valeurs saisies
# ---------------------------------------------------------------------------
def parse_grade(value: Any, maximum: int = 3) -> Optional[int]:
    """Note lue dans une cellule.

    Args:
        value: contenu de la cellule (« 2 », « 2.0 » et « 2,0 » acceptés).
        maximum: note maximale de l'échelle.

    Returns:
        La note entière, ou ``None`` si la cellule est vide.

    Raises:
        ValueError: valeur illisible ou hors échelle (jamais ignorée).
    """
    text = str(value if value is not None else "").strip().replace(",", ".")
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        raise ValueError(f"note illisible : {value!r}") from None
    if not math.isfinite(number) or number != int(number) or not 0 <= int(number) <= maximum:
        raise ValueError(f"note hors échelle (0 à {maximum}) : {value!r}")
    return int(number)


def parse_preference(value: Any) -> Optional[str]:
    """Préférence A/B lue dans une cellule.

    Args:
        value: « A », « B », « égal » (aussi « egal », « e », « = »), casse indifférente.

    Returns:
        ``"A"``, ``"B"`` ou ``"égal"``, ou ``None`` si la cellule est vide.

    Raises:
        ValueError: valeur illisible.
    """
    text = str(value if value is not None else "").strip()
    if text == "":
        return None
    folded = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
    if folded == "a":
        return "A"
    if folded == "b":
        return "B"
    if folded in ("egal", "e", "=", "egaux", "equal"):
        return "égal"
    raise ValueError(f"préférence illisible (A, B ou égal) : {value!r}")


# ---------------------------------------------------------------------------
# Hasard à graine fixe
# ---------------------------------------------------------------------------
def seeded_shuffle(values: Iterable[Any], seed: int, salt: str) -> List[Any]:
    """Copie mélangée par un générateur initialisé sur ``(graine, sel)``.

    Args:
        values: valeurs (l'ordre d'entrée doit être lui-même déterministe).
        seed: graine.
        salt: sel propre à l'usage (fiche, document…).

    Returns:
        La liste mélangée.
    """
    out = list(values)
    random.Random(f"{seed}:{salt}").shuffle(out)
    return out


def second_sample(ids: Sequence[str], ratio: float, seed: int, salt: str) -> List[str]:
    """Échantillon du second jugement, dans un autre ordre.

    ``ceil(ratio × n)`` identifiants (au moins un s'il y en a), tirés à graine
    fixe parmi les identifiants triés, puis remélangés avec un autre sel.

    Args:
        ids: identifiants du premier jugement.
        ratio: part à rejuger (0 : aucun second jugement).
        seed: graine.
        salt: sel propre à la fiche.

    Returns:
        Les identifiants retenus, dans l'ordre de présentation du second jugement.
    """
    pool = sorted(set(ids))
    if not pool or ratio <= 0:
        return []
    k = min(len(pool), max(1, math.ceil(ratio * len(pool) - 1e-9)))
    chosen = random.Random(f"{seed}:{salt}:echantillon").sample(pool, k)
    return seeded_shuffle(sorted(chosen), seed, f"{salt}:ordre-second")


def id_width(count: int) -> int:
    """Nombre de chiffres des identifiants numérotés (3 au moins).

    Args:
        count: nombre d'identifiants.

    Returns:
        La largeur de remplissage.
    """
    return max(3, len(str(max(count, 1))))


def sheet_fingerprint(prefix: str, parts: Iterable[Any]) -> str:
    """Identifiant de fiche dérivé de son contenu (clé du stockage local).

    Args:
        prefix: nom de la fiche.
        parts: éléments qui définissent la fiche (identifiants, ordre…).

    Returns:
        ``<prefix>-<10 caractères hexadécimaux>``.
    """
    digest = hashlib.sha256(json.dumps(list(parts), ensure_ascii=False, sort_keys=True,
                                       default=str).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:10]}"


# ---------------------------------------------------------------------------
# Accord inter-juges
# ---------------------------------------------------------------------------
def weighted_kappa(pairs: Sequence[Tuple[int, int]], categories: Sequence[int] = (0, 1, 2, 3),
                   weights: str = "quadratic") -> Optional[float]:
    """Kappa de Cohen pondéré entre deux séries de notes.

    Poids d'accord ``w_ij = 1 - |i - j|^p / (K - 1)^p`` (``p = 2``
    quadratique, ``p = 1`` linéaire ; ``"none"`` : 1 si ``i = j``, sinon 0).
    ``kappa = (Po - Pe) / (1 - Pe)`` avec ``Po = Σ w_ij O_ij / n`` et
    ``Pe = Σ w_ij (ligne_i × colonne_j) / n²``.

    Args:
        pairs: couples ``(premier jugement, second jugement)``.
        categories: valeurs de l'échelle, dans l'ordre.
        weights: ``"quadratic"``, ``"linear"`` ou ``"none"``.

    Returns:
        Le kappa, ou ``None`` si moins de deux couples ou aucune variabilité
        (``Pe = 1``).

    Raises:
        ValueError: note hors des catégories.
    """
    if len(pairs) < 2:
        return None
    cats = list(categories)
    index = {c: i for i, c in enumerate(cats)}
    size = len(cats)
    observed = [[0] * size for _ in range(size)]
    for a, b in pairs:
        if a not in index or b not in index:
            raise ValueError(f"note hors échelle dans le calcul du kappa : {(a, b)!r}")
        observed[index[a]][index[b]] += 1
    n = float(len(pairs))
    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(size)) for j in range(size)]

    def weight(i: int, j: int) -> float:
        """Poids d'accord entre les catégories ``i`` et ``j``."""
        if weights == "none" or size == 1:
            return 1.0 if i == j else 0.0
        power = 2 if weights == "quadratic" else 1
        return 1.0 - (abs(i - j) ** power) / ((size - 1) ** power)

    po = sum(weight(i, j) * observed[i][j] for i in range(size) for j in range(size)) / n
    pe = sum(weight(i, j) * rows[i] * cols[j] for i in range(size) for j in range(size)) / (n * n)
    if pe >= 1.0 - 1e-12:
        return None
    return (po - pe) / (1.0 - pe)


# ---------------------------------------------------------------------------
# Formats d'affichage
# ---------------------------------------------------------------------------
def fmt(value: Optional[float], digits: int = 3) -> str:
    """Nombre formaté à la française (« — » si indéfini).

    Args:
        value: nombre ou ``None``.
        digits: décimales.

    Returns:
        La chaîne (virgule décimale).
    """
    if value is None:
        return "—"
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: Optional[float]) -> str:
    """Pourcentage formaté à la française (« — » si indéfini).

    Args:
        value: proportion entre 0 et 1, ou ``None``.

    Returns:
        La chaîne (« 12,5 % »).
    """
    if value is None:
        return "—"
    return f"{100 * value:.1f} %".replace(".", ",")


def duration_label(seconds: float) -> str:
    """Durée lisible (« 1 h 36 min », « 12 min »).

    Args:
        seconds: durée en secondes.

    Returns:
        Le libellé arrondi à la minute.
    """
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{max(minutes, 1)} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


# ---------------------------------------------------------------------------
# Page HTML autonome
# ---------------------------------------------------------------------------
_CSS = """
:root{--bg:#f7f6f2;--fg:#1c1c1a;--muted:#5d5b55;--card:#ffffff;--line:#d9d5cb;--accent:#1f5f8b;
--accent-fg:#ffffff;--done:#2e7d4f;--warn:#9a5b00;--sel:#e6eef6;--text-bg:#fbfaf7}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161614;--fg:#ecebe6;
--muted:#a8a59c;--card:#201f1c;--line:#3a3833;--accent:#7db3db;--accent-fg:#10202c;--done:#6fcf97;
--warn:#f2b35b;--sel:#233340;--text-bg:#1b1a18}}
:root[data-theme="dark"]{--bg:#161614;--fg:#ecebe6;--muted:#a8a59c;--card:#201f1c;--line:#3a3833;
--accent:#7db3db;--accent-fg:#10202c;--done:#6fcf97;--warn:#f2b35b;--sel:#233340;--text-bg:#1b1a18}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header.top{padding:16px;border-bottom:1px solid var(--line)}
h1{font-size:1.3rem;margin:0 0 6px}
.help{color:var(--muted);font-size:.92rem;margin:4px 0}
.scale{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0;padding:0;list-style:none;font-size:.92rem}
.scale b{font-variant-numeric:tabular-nums}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:10px}
button,.filebtn{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);
border-radius:6px;padding:5px 10px;cursor:pointer}
button:focus-visible,.filebtn:focus-within,textarea:focus-visible{outline:2px solid var(--accent);outline-offset:1px}
button.primary{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
.progress{flex:1 1 220px;min-width:160px}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden}
.bar>div{height:100%;width:0;background:var(--done)}
.confirm{display:none;gap:6px;align-items:center;flex-wrap:wrap;color:var(--warn)}
.confirm.open{display:flex}
#status{min-height:1.3em}
main{display:grid;grid-template-columns:230px minmax(0,1fr);min-height:70vh}
nav#list{border-right:1px solid var(--line);padding:10px;overflow:auto;max-height:100vh;position:sticky;top:0}
nav#list h2{font-size:.85rem;color:var(--muted);margin:10px 0 4px;text-transform:none}
nav#list ol{list-style:none;margin:0;padding:0;display:flex;flex-wrap:wrap;gap:4px}
nav#list button{padding:2px 6px;font-size:.82rem;font-variant-numeric:tabular-nums}
nav#list button.done{border-color:var(--done);color:var(--done)}
nav#list button[aria-current="true"]{background:var(--sel);font-weight:600}
section#item{padding:16px;min-width:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;max-width:1500px}
.card.done{border-left:5px solid var(--done)}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 14px;margin-bottom:8px}
.head h2{font-size:1.05rem;margin:0}
.badge{font-size:.8rem;border:1px solid var(--line);border-radius:10px;padding:1px 8px;color:var(--muted)}
.body.with-image{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;align-items:start}
figure{margin:0}
figure img{width:100%;height:auto;border:1px solid var(--line);background:#fff;cursor:zoom-in}
figure.zoom img{cursor:zoom-out}
.body.with-image.zoomed{grid-template-columns:1fr}
figcaption{font-size:.85rem;color:var(--muted)}
.clip{margin:0 0 10px}.clip audio{width:100%}
.block{margin:0 0 10px}
.block h3{font-size:.85rem;color:var(--muted);margin:0 0 3px;font-weight:600}
.txt{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--text-bg);border:1px solid var(--line);
border-radius:6px;padding:8px 10px;max-height:70vh;overflow:auto}
.versions{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px}
.grades{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.grades button[aria-pressed="true"]{background:var(--accent);color:var(--accent-fg);border-color:var(--accent)}
.grades small{color:inherit;opacity:.8}
label.comment{display:block;font-size:.9rem;color:var(--muted)}
textarea{width:100%;min-height:3.2em;font:inherit;color:var(--fg);background:var(--bg);
border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin-top:3px}
.nav{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap}
.missing{color:var(--warn)}
@media (max-width:860px){main{grid-template-columns:1fr}nav#list{position:static;max-height:30vh;
border-right:0;border-bottom:1px solid var(--line)}.body.with-image,.versions{grid-template-columns:1fr}
header.top,section#item{padding:12px 16px}}
"""

_JS = r"""
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('sheet-data').textContent);
  var ITEMS = DATA.items;
  var KEY = 'ragpy-albert-eval:' + DATA.sheet_id + ':' + DATA.seed;
  var PREF_KEYS = {a: 'A', b: 'B', e: 'égal', '=': 'égal'};
  var store = {};
  var cur = 0;
  try {
    var raw = window.localStorage.getItem(KEY);
    if (raw) {
      var parsed = JSON.parse(raw);
      if (parsed && typeof parsed === 'object') { store = parsed; }
    }
    var pos = parseInt(window.localStorage.getItem(KEY + ':position') || '0', 10);
    if (pos >= 0 && pos < ITEMS.length) { cur = pos; }
  } catch (e) {
    store = {};
  }
  var byId = {};
  ITEMS.forEach(function (it, i) { byId[it.id] = i; });
  var navButtons = [];
  var nav = document.getElementById('list');
  var view = document.getElementById('item');
  var statusEl = document.getElementById('status');

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined && text !== null) { e.textContent = String(text); }
    return e;
  }
  function state(id) {
    var s = store[id];
    if (!s || typeof s !== 'object') { s = {note: '', preference: '', commentaire: ''}; }
    return s;
  }
  function isDone(it) {
    var s = state(it.id);
    return it.input === 'preference' ? !!s.preference : (s.note !== undefined && s.note !== '');
  }
  function save() {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(store));
      window.localStorage.setItem(KEY + ':position', String(cur));
    } catch (e) {
      say('Sauvegarde dans le navigateur indisponible : exportez le CSV régulièrement.');
    }
  }
  function say(msg) { statusEl.textContent = msg; }
  function markLabel(it) {
    var s = state(it.id);
    var v = it.input === 'preference' ? s.preference : s.note;
    return it.id + (v ? ' · ' + v : '');
  }
  function buildNav() {
    nav.textContent = '';
    navButtons = [];
    var section = null, list = null;
    ITEMS.forEach(function (it, i) {
      if (it.section !== section) {
        section = it.section;
        nav.appendChild(el('h2', '', section));
        list = el('ol');
        nav.appendChild(list);
      }
      var li = el('li');
      var b = el('button', '', markLabel(it));
      b.type = 'button';
      b.addEventListener('click', function () { go(i); });
      li.appendChild(b);
      list.appendChild(li);
      navButtons.push(b);
    });
  }
  function refreshNav(i) {
    var b = navButtons[i];
    if (!b) { return; }
    b.textContent = markLabel(ITEMS[i]);
    b.classList.toggle('done', isDone(ITEMS[i]));
    b.setAttribute('aria-current', i === cur ? 'true' : 'false');
  }
  function refreshProgress() {
    var done = 0;
    ITEMS.forEach(function (it) { if (isDone(it)) { done += 1; } });
    var total = ITEMS.length;
    document.getElementById('progress-text').textContent =
      done + ' / ' + total + ' éléments jugés';
    document.getElementById('progress-fill').style.width = (total ? (100 * done / total) : 0) + '%';
  }
  function setValue(field, value, advance) {
    var it = ITEMS[cur];
    var s = state(it.id);
    s[field] = value;
    store[it.id] = s;
    save();
    refreshNav(cur);
    refreshProgress();
    if (advance && cur < ITEMS.length - 1) { go(cur + 1); } else { render(); }
  }
  function choices(it) {
    return it.input === 'preference' ? DATA.preferences : DATA.scale;
  }
  function render() {
    var it = ITEMS[cur];
    var s = state(it.id);
    var old = document.getElementById('clip');
    if (old && old.pause) { try { old.pause(); } catch (e) { /* lecture d\u00e9j\u00e0 arr\u00eat\u00e9e */ } }
    view.textContent = '';
    var card = el('article', 'card' + (isDone(it) ? ' done' : ''));
    var head = el('div', 'head');
    head.appendChild(el('h2', '', 'Élément ' + it.id));
    head.appendChild(el('span', 'badge', it.section));
    head.appendChild(el('span', 'help', (cur + 1) + ' sur ' + ITEMS.length));
    card.appendChild(head);
    var body = el('div', 'body' + (it.image ? ' with-image' : ''));
    if (it.image && DATA.images[it.image]) {
      var fig = el('figure');
      var img = document.createElement('img');
      img.alt = it.image_alt || 'Image de la page';
      img.src = DATA.images[it.image];
      img.addEventListener('click', function () {
        fig.classList.toggle('zoom');
        body.classList.toggle('zoomed');
      });
      fig.appendChild(img);
      fig.appendChild(el('figcaption', '', 'Cliquer l’image pour l’agrandir ou la réduire.'));
      body.appendChild(fig);
    } else if (it.image) {
      body.appendChild(el('p', 'missing', 'Image omise (taille maximale de la fiche atteinte).'));
    }
    var texts = el('div', 'texts');
    if (it.audio) {
      var clip = el('div', 'clip');
      if (DATA.audio && DATA.audio[it.audio]) {
        var au = document.createElement('audio');
        au.controls = true;
        au.preload = 'metadata';
        au.id = 'clip';
        au.src = DATA.audio[it.audio];
        clip.appendChild(au);
        clip.appendChild(el('p', 'help', 'Touche p : \u00e9couter ou mettre en pause ; r : r\u00e9\u00e9couter depuis le d\u00e9but.'));
      } else {
        clip.appendChild(el('p', 'missing', 'Extrait audio indisponible.'));
      }
      texts.appendChild(clip);
    }
    var versions = null;
    it.blocks.forEach(function (bl) {
      var block = el('section', 'block ' + (bl.cls || ''));
      block.appendChild(el('h3', '', bl.label));
      block.appendChild(el('div', 'txt', bl.text === '' ? '(vide)' : bl.text));
      if (bl.cls === 'version') {
        if (!versions) { versions = el('div', 'versions'); texts.appendChild(versions); }
        versions.appendChild(block);
      } else {
        texts.appendChild(block);
      }
    });
    body.appendChild(texts);
    card.appendChild(body);
    var field = it.input === 'preference' ? 'preference' : 'note';
    var grades = el('div', 'grades');
    grades.setAttribute('role', 'group');
    grades.setAttribute('aria-label', it.input === 'preference' ? 'Préférence' : 'Note');
    choices(it).forEach(function (c) {
      var b = el('button');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(s[field] === String(c[0])));
      b.appendChild(el('b', '', c[1]));
      if (c[2]) { b.appendChild(el('small', '', ' — ' + c[2])); }
      b.title = 'Touche ' + c[3];
      b.addEventListener('click', function () { setValue(field, String(c[0]), true); });
      grades.appendChild(b);
    });
    card.appendChild(grades);
    var lab = el('label', 'comment', 'Commentaire (facultatif, touche c ; Échap pour revenir au clavier)');
    var ta = document.createElement('textarea');
    ta.id = 'comment';
    ta.value = s.commentaire || '';
    ta.addEventListener('input', function () {
      var st = state(it.id);
      st.commentaire = ta.value;
      store[it.id] = st;
      save();
    });
    lab.appendChild(ta);
    card.appendChild(lab);
    var navb = el('div', 'nav');
    var prev = el('button', '', '← Précédent');
    prev.type = 'button';
    prev.disabled = cur === 0;
    prev.addEventListener('click', function () { go(cur - 1); });
    var next = el('button', '', 'Suivant →');
    next.type = 'button';
    next.disabled = cur === ITEMS.length - 1;
    next.addEventListener('click', function () { go(cur + 1); });
    navb.appendChild(prev);
    navb.appendChild(next);
    card.appendChild(navb);
    view.appendChild(card);
  }
  function go(i) {
    if (i < 0 || i >= ITEMS.length) { return; }
    var old = cur;
    cur = i;
    refreshNav(old);
    refreshNav(cur);
    save();
    render();
    if (navButtons[cur] && navButtons[cur].scrollIntoView) { navButtons[cur].scrollIntoView({block: 'nearest'}); }
    window.scrollTo(0, 0);
  }
  function nextUndone() {
    for (var step = 1; step <= ITEMS.length; step += 1) {
      var i = (cur + step) % ITEMS.length;
      if (!isDone(ITEMS[i])) { go(i); return; }
    }
    if (!isDone(ITEMS[cur])) { return; }
    say('Tous les éléments sont jugés : exportez le CSV.');
  }
  function csvCell(v) {
    v = String(v === undefined || v === null ? '' : v);
    return /[;"\r\n]/.test(v) || /^\s|\s$/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  }
  function exportCsv() {
    var cols = DATA.columns;
    var lines = [cols.map(csvCell).join(';')];
    ITEMS.forEach(function (it) {
      var s = state(it.id);
      var row = {};
      Object.keys(it.row).forEach(function (k) { row[k] = it.row[k]; });
      row.note = s.note || '';
      row.preference = s.preference || '';
      row.commentaire = (s.commentaire || '').replace(/\r?\n/g, ' ');
      lines.push(cols.map(function (c) { return csvCell(row[c]); }).join(';'));
    });
    var blob = new Blob(['﻿' + lines.join('\r\n') + '\r\n'], {type: 'text/csv;charset=utf-8'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = DATA.file_stem + '_rempli.csv';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1500);
    say('Fichier ' + DATA.file_stem + '_rempli.csv produit : placez-le dans le dossier de la campagne.');
  }
  function parseCsv(t) {
    var head = t.split(/\r?\n/)[0] || '';
    var d = (head.match(/;/g) || []).length >= (head.match(/,/g) || []).length ? ';' : ',';
    var out = [], row = [], cell = '', q = false;
    for (var i = 0; i < t.length; i += 1) {
      var ch = t[i];
      if (q) {
        if (ch === '"') { if (t[i + 1] === '"') { cell += '"'; i += 1; } else { q = false; } } else { cell += ch; }
      } else if (ch === '"') { q = true; }
      else if (ch === d) { row.push(cell); cell = ''; }
      else if (ch === '\n') { row.push(cell.replace(/\r$/, '')); out.push(row); row = []; cell = ''; }
      else { cell += ch; }
    }
    if (cell !== '' || row.length) { row.push(cell); out.push(row); }
    return out;
  }
  function importCsv(file) {
    var rd = new FileReader();
    rd.onload = function () {
      var rows = parseCsv(String(rd.result).replace(/^﻿/, ''));
      if (!rows.length) { say('Fichier vide.'); return; }
      var h = rows[0].map(function (x) { return x.trim().toLowerCase(); });
      var ii = h.indexOf('item_id'), ni = h.indexOf('note'), pi = h.indexOf('preference'), ci = h.indexOf('commentaire');
      if (ii < 0) { say('Colonne item_id absente : import impossible.'); return; }
      var n = 0, bad = 0;
      rows.slice(1).forEach(function (r) {
        var id = (r[ii] || '').trim();
        if (!(id in byId)) { return; }
        var s = state(id), changed = false;
        var note = ni >= 0 ? (r[ni] || '').trim() : '';
        if (note !== '') { if (/^[0-3]$/.test(note)) { s.note = note; changed = true; } else { bad += 1; } }
        var pref = pi >= 0 ? (r[pi] || '').trim() : '';
        if (pref !== '') {
          var p = pref.toLowerCase();
          if (p === 'a' || p === 'b') { s.preference = p.toUpperCase(); changed = true; }
          else if (p === 'égal' || p === 'egal' || p === 'e' || p === '=') { s.preference = 'égal'; changed = true; }
          else { bad += 1; }
        }
        if (ci >= 0 && (r[ci] || '').trim() !== '') { s.commentaire = r[ci].trim(); changed = true; }
        if (changed) { store[id] = s; n += 1; }
      });
      save();
      buildNav();
      ITEMS.forEach(function (_it, i) { refreshNav(i); });
      refreshProgress();
      render();
      say(n + ' ligne(s) importée(s)' + (bad ? ', ' + bad + ' valeur(s) illisible(s) ignorée(s)' : '') + '.');
    };
    rd.readAsText(file, 'utf-8');
  }
  document.getElementById('export').addEventListener('click', exportCsv);
  document.getElementById('next-undone').addEventListener('click', nextUndone);
  document.getElementById('import').addEventListener('change', function (ev) {
    var f = ev.target.files && ev.target.files[0];
    if (f) { importCsv(f); }
    ev.target.value = '';
  });
  var confirmBox = document.getElementById('confirm');
  document.getElementById('reset').addEventListener('click', function () { confirmBox.classList.add('open'); });
  document.getElementById('reset-no').addEventListener('click', function () { confirmBox.classList.remove('open'); });
  document.getElementById('reset-yes').addEventListener('click', function () {
    store = {};
    cur = 0;
    try {
      window.localStorage.removeItem(KEY);
      window.localStorage.removeItem(KEY + ':position');
    } catch (e) { /* stockage indisponible : rien à effacer */ }
    confirmBox.classList.remove('open');
    buildNav();
    ITEMS.forEach(function (_it, i) { refreshNav(i); });
    refreshProgress();
    render();
    say('Jugements effacés de ce navigateur.');
  });
  document.addEventListener('keydown', function (ev) {
    var t = ev.target;
    var typing = t && (t.tagName === 'TEXTAREA' || (t.tagName === 'INPUT' && t.type !== 'file') || t.isContentEditable);
    if (typing) {
      if (ev.key === 'Escape') { t.blur(); ev.preventDefault(); }
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) { return; }
    var it = ITEMS[cur];
    var key = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    if (it.input !== 'preference' && /^[0-3]$/.test(key)) { setValue('note', key, true); ev.preventDefault(); return; }
    if (it.input === 'preference' && PREF_KEYS[key]) { setValue('preference', PREF_KEYS[key], true); ev.preventDefault(); return; }
    if (key === 'ArrowRight' || key === 'j') { go(cur + 1); ev.preventDefault(); return; }
    if (key === 'ArrowLeft' || key === 'k') { go(cur - 1); ev.preventDefault(); return; }
    if (key === 'n') { nextUndone(); ev.preventDefault(); return; }
    if (key === 'p' || key === 'r') {
      var au = document.getElementById('clip');
      if (au) {
        try {
          if (key === 'r') { au.currentTime = 0; au.play(); } else if (au.paused) { au.play(); } else { au.pause(); }
        } catch (e) { say('Lecture impossible dans ce navigateur.'); }
        ev.preventDefault();
      }
      return;
    }
    if (key === 'c') {
      var ta = document.getElementById('comment');
      if (ta) { ta.focus(); ev.preventDefault(); }
    }
  });
  buildNav();
  ITEMS.forEach(function (_it, i) { refreshNav(i); });
  refreshProgress();
  if (ITEMS.length) { render(); } else { view.textContent = 'Aucun élément à juger.'; }
})();
"""


def _json_for_script(payload: Mapping[str, Any]) -> str:
    """JSON sûr à l'intérieur d'une balise ``<script type="application/json">``.

    Args:
        payload: données de la fiche.

    Returns:
        Le JSON avec ``<``, ``>`` et ``&`` échappés en ``\\u00XX``.
    """
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")


def render_sheet_html(*, title: str, intro: Sequence[str], rules: Sequence[str], sheet_id: str, seed: int,
                      file_stem: str, columns: Sequence[str], items: Sequence[Mapping[str, Any]],
                      scale: Sequence[Tuple[int, str, str]], scale_title: str,
                      preferences: Optional[Sequence[Tuple[str, str, str]]] = None,
                      images: Optional[Mapping[str, str]] = None,
                      audio: Optional[Mapping[str, str]] = None) -> str:
    """Page HTML autonome de jugement (aucune ressource externe).

    Tout texte venu des données passe par le JSON embarqué, puis par
    ``textContent`` côté navigateur ; les textes fixes passent par
    ``html.escape``.

    Args:
        title: titre de la page.
        intro: paragraphes d'introduction (texte brut).
        rules: règles de jugement (texte brut, une par puce).
        sheet_id: identifiant de la fiche (clé du stockage local, avec la graine).
        seed: graine de la fiche.
        file_stem: nom de base du CSV exporté (``<file_stem>_rempli.csv``).
        columns: colonnes du CSV exporté.
        items: éléments ``{id, section, input, row, blocks, image?, image_alt?, audio?}``.
        scale: échelle des notes ``(valeur, libellé, définition)``.
        scale_title: intitulé de l'échelle.
        preferences: choix de la comparaison A/B (si la fiche en contient).
        images: images ``{id: URI data:}`` référencées par les éléments.
        audio: extraits audio ``{id: URI data:}`` référencés par les éléments.

    Returns:
        Le document HTML complet.
    """
    esc = html.escape
    scale_keys = [(str(v), f"{v} {label}", definition, str(v)) for v, label, definition in scale]
    pref_keys = {"A": "a", "B": "b", "égal": "e"}
    prefs = [(v, label, definition, pref_keys.get(v, "")) for v, label, definition in (preferences or ())]
    payload = {
        "sheet_id": sheet_id, "seed": seed, "file_stem": file_stem, "columns": list(columns),
        "scale": scale_keys, "preferences": prefs, "images": dict(images or {}), "audio": dict(audio or {}),
        "items": list(items),
    }
    scale_html = "".join(f"<li><b>{v}</b> {esc(label)} : {esc(definition)}</li>" for v, label, definition in scale)
    pref_html = ""
    if preferences:
        pref_html = ('<p class="help"><b>Comparaison A/B</b> (touches a, b, e) : '
                     + " · ".join(f"<b>{esc(v)}</b> {esc(definition)}" for v, _label, definition in preferences)
                     + "</p>")
    intro_html = "".join(f'<p class="help">{esc(p)}</p>' for p in intro)
    rules_html = "".join(f"<li>{esc(r)}</li>" for r in rules)
    keys = ("Clavier : 0 à 3 pour noter (passage automatique à l'élément suivant)"
            + (", a, b ou e pour la comparaison A/B" if preferences else "")
            + (" ; p pour écouter l'extrait ou le mettre en pause, r pour le réécouter" if audio else "")
            + " ; ← → ou j k pour naviguer ; n pour le prochain élément non jugé ; c pour écrire un commentaire, "
              "Échap pour en sortir.")
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>{esc(title)}</title>
<style>{_CSS}</style></head>
<body>
<header class="top">
<h1>{esc(title)}</h1>
{intro_html}
<p class="help"><b>{esc(scale_title)}</b></p>
<ul class="scale">{scale_html}</ul>
{pref_html}
<ul class="help">{rules_html}</ul>
<p class="help">{esc(keys)} Les jugements sont gardés dans ce navigateur ; « Exporter le CSV » produit le fichier à rendre.</p>
<div class="toolbar">
<div class="progress"><div class="bar"><div id="progress-fill"></div></div><span id="progress-text" class="help"></span></div>
<button type="button" id="next-undone">Prochain non jugé</button>
<button type="button" id="export" class="primary">Exporter le CSV</button>
<label class="filebtn">Importer un CSV <input id="import" type="file" accept=".csv,text/csv" hidden></label>
<button type="button" id="reset">Réinitialiser</button>
<span id="confirm" class="confirm" role="group" aria-label="Confirmation">Effacer tous les jugements de cette fiche gardés dans ce navigateur ? Exportez d'abord le CSV si besoin.
<button type="button" id="reset-yes">Oui, tout effacer</button><button type="button" id="reset-no">Annuler</button></span>
</div>
<p id="status" class="help" role="status" aria-live="polite"></p>
</header>
<main><nav id="list" aria-label="Éléments à juger"></nav><section id="item" aria-live="polite"></section></main>
<script type="application/json" id="sheet-data">{_json_for_script(payload)}</script>
<script>{_JS}</script>
</body></html>
"""
