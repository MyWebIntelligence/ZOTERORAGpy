"""Registre unique des variables d'environnement de RAGpy.

Sprint « configuration unifiée » (``.claude/tasks/SPRINT_config_unifiee.md``),
lot L1. Ce module est la **seule source de vérité** sur les variables : l'ordre
de ``LAYOUT`` est l'ordre du ``.env``, de ``.env.example`` et de la future page
« Paramètres » ; ``scripts/env_tool.py`` en tire ``.env.example`` et range le
``.env``.

Le module n'utilise que la bibliothèque standard : il est importable par
l'application (``scripts.rad_settings.registry``) comme par les scripts lancés
avec ``scripts/`` sur ``sys.path`` (``rad_settings.registry``).

Vocabulaire :

* ``default`` : défaut **actuel du code** (chaîne exacte ; ``""`` = aucun). Il
  documente le comportement d'aujourd'hui ; il n'est jamais lu à l'exécution
  (le code garde ses propres défauts jusqu'au lot L3).
* ``example`` : valeur montrée dans ``.env.example`` (``None`` = ``default``).
* ``render`` : rendu dans ``.env.example`` : ``active`` (ligne ``NOM=valeur``),
  ``commented`` (``# NOM=valeur``) ou ``hidden`` (absente).
* ``status`` : ``active`` (lue par le code), ``planned`` (introduite par un lot
  suivant : déclarée ici pour l'ordre et l'interface, pas encore lue).
* ``scope`` : ``user`` (déclarable par chaque utilisateur), ``server``
  (administrateur), ``deploy`` (déploiement, lue au démarrage), ``build``
  (construction de l'image Docker).
* ``apply`` : ce qu'exige une modification : ``hot`` (rien : le ``.env`` est
  relu quand il change, lot L3), ``restart`` (redémarrer le serveur : valeur lue
  à l'import d'un module ou au démarrage), ``recreate`` (``docker compose up -d``),
  ``rebuild`` (``docker compose up -d --build``).
* ``locked`` : raison pour laquelle l'interface affiche la variable sans
  permettre de la modifier (vide = modifiable).

Généré au lot L1 à partir du ``.env.example`` du 2026-10-02 (descriptions
reprises de ses commentaires), puis maintenu à la main.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional, Tuple

# Natures de valeur
SECRET = "secret"
URL = "url"
SERVER = "server"
MODEL = "model"
BOOL = "bool"
INT = "int"
FLOAT = "float"
ENUM = "enum"
LIST = "list"
STR = "str"
PATH = "path"
KINDS = (SECRET, URL, SERVER, MODEL, BOOL, INT, FLOAT, ENUM, LIST, STR, PATH)

# Portées
USER = "user"
SERVER_SCOPE = "server"
DEPLOY = "deploy"
BUILD = "build"
SCOPES = (USER, SERVER_SCOPE, DEPLOY, BUILD)

# Effet d'une modification
HOT = "hot"
RESTART = "restart"
RECREATE = "recreate"
REBUILD = "rebuild"
APPLIES = (HOT, RESTART, RECREATE, REBUILD)

# Statuts
ACTIVE = "active"
PLANNED = "planned"
STATUSES = (ACTIVE, PLANNED)

RENDERS = ("active", "commented", "hidden")
BOOL_STYLES = ("01", "truefalse", "TRUEFALSE")
FAMILIES = ("chat", "ocr", "embedding", "audio")


@dataclass(frozen=True)
class Setting:
    """Une variable d'environnement et ses métadonnées.

    Attributes:
        name: Nom de la variable (``MISTRAL_API_KEY``).
        kind: Nature de la valeur (``SECRET``, ``URL``, ``BOOL``…).
        default: Défaut actuel du code, chaîne exacte (``""`` = aucun).
        doc: Description en français (lignes séparées par ``\\n``), reprise en
            commentaire dans ``.env.example``.
        example: Valeur montrée dans ``.env.example`` (``None`` = ``default``).
        render: ``active``, ``commented`` ou ``hidden`` dans ``.env.example``.
        status: ``active`` (lue par le code) ou ``planned`` (lot suivant).
        scope: ``user``, ``server``, ``deploy`` ou ``build``.
        apply: ``hot`` (prise en compte à la requête ou au traitement suivant),
            ``restart``, ``recreate`` ou ``rebuild``.
        choices: Valeurs admises pour une ``ENUM``.
        bool_style: Écriture canonique d'un booléen (``01``, ``truefalse``,
            ``TRUEFALSE``).
        family: Famille d'un modèle ou d'un serveur (``chat``, ``ocr``…).
        locked: Raison du verrouillage dans l'interface (vide = modifiable).
        default_rule: Défaut calculé par le code quand il n'est pas une
            constante (« nombre de CPU − 1 »).
        replaced_by: Variables qui la remplaceront (lots suivants).
        block: Clé du bloc (``"1"``), renseignée par l'aplatissement.
        subblock: Clé du sous-bloc (``"1.2"``), renseignée par l'aplatissement.
    """

    name: str
    kind: str
    default: str = ""
    doc: str = ""
    example: Optional[str] = None
    render: str = "active"
    status: str = ACTIVE
    scope: str = SERVER_SCOPE
    apply: str = HOT
    choices: Tuple[str, ...] = ()
    bool_style: str = ""
    family: str = ""
    locked: str = ""
    default_rule: str = ""
    replaced_by: Tuple[str, ...] = ()
    block: str = ""
    subblock: str = ""

    @property
    def example_value(self) -> str:
        """Valeur écrite dans ``.env.example`` (``example``, sinon ``default``)."""
        return self.default if self.example is None else self.example

    @property
    def is_secret(self) -> bool:
        """Vrai pour une clé ou un mot de passe (affichage masqué, écriture seule)."""
        return self.kind == SECRET


@dataclass(frozen=True)
class SubBlock:
    """Sous-bloc numéroté (``# --- 1.2 Mistral ---``)."""

    key: str
    title: str
    settings: Tuple[Setting, ...]
    doc: str = ""


@dataclass(frozen=True)
class Block:
    """Bloc numéroté (``# ===== 1. SERVEURS D'API… =====``)."""

    key: str
    title: str
    subblocks: Tuple[SubBlock, ...]
    doc: str = ""


def S(name: str, kind: str, **fields) -> Setting:  # noqa: N802 - écriture compacte du registre
    """Construit un ``Setting`` (raccourci d'écriture du registre)."""
    return Setting(name, kind, **fields)


LAYOUT: Tuple[Block, ...] = (
    Block('1', "SERVEURS D'API ET SERVICES EXTERNES : ADRESSES ET CLÉS", doc="Un service (bloc 2) désigne son serveur par son ADRESSE ; la clé envoyée est celle\ndu sous-bloc dont l'adresse correspond (une adresse non déclarée ici est refusée).\nAdresses vérifiées et modèles réputés : README.md, section « Serveurs d'API et modèles ».", subblocks=(
        SubBlock('1.1', 'OpenRouter', settings=(
            S('OPENROUTER_API_BASE_URL', URL, default='', example='https://openrouter.ai/api/v1', doc="Adresse de l'API OpenRouter."),
            S('OPENROUTER_API_KEY', SECRET, default='', example='sk-or-v1-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', scope=USER, doc='Alternative économique (~75 % moins chère pour le recodage).'),
        )),
        SubBlock('1.2', 'Mistral', settings=(
            S('MISTRAL_API_BASE_URL', URL, default='https://api.mistral.ai', example='https://api.mistral.ai/v1', scope=USER, doc="Adresse de l'API Mistral (OCR et modèles de langage), avec ou sans /v1."),
            S('MISTRAL_API_KEY', SECRET, default='', example='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', scope=USER, doc='OCR Mistral, premier maillon de la chaîne OCR par défaut. Un 401 peut signifier le\nplafond de dépense mensuel atteint (console.mistral.ai → Limits), pas seulement\nune clé invalide.'),
        )),
        SubBlock('1.3', 'OpenAI', settings=(
            S('OPENAI_API_BASE_URL', URL, default='', example='https://api.openai.com/v1', render='commented', doc="Adresse de l'API OpenAI."),
            S('OPENAI_API_KEY', SECRET, default='', example='sk-proj-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', render='commented', scope=USER, doc='Embeddings text-embedding-3-large (3072 d) et modèles OpenAI (recodage, notes).\nNécessaire tant que les embeddings restent sur OpenAI.'),
        )),
        SubBlock('1.4', 'Albert (DINUM)', settings=(
            S('ALBERT_BASE_URL', URL, default='https://albert.api.etalab.gouv.fr/v1', render='commented', locked='liste blanche : la clé Albert de chaque utilisateur partirait vers une autre adresse', doc='URL de base, configuration serveur uniquement : https, /v1 garanti, hôte en\nliste blanche ; vide = défaut.'),
            S('ALBERT_API_KEY', SECRET, default='', render='commented', scope=USER, doc='Clé Bearer du compte Albert (identifiant albert_api_key). Repli .env réservé aux\nadmins ; un non-admin saisit sa propre clé dans Paramètres > Mes Identifiants.\nJamais renvoyée en clair (masquée par le serveur).'),
            S('ALBERT_ALLOW_CUSTOM_HOST', BOOL, default='0', render='commented', bool_style='01', locked='liste blanche : la clé Albert de chaque utilisateur partirait vers une autre adresse', doc='1 = autorise un OpenGateLLM auto-hébergé hors liste blanche (https reste imposé).'),
        )),
        SubBlock('1.5', 'Anthropic', settings=(
            S('ANTHROPIC_API_BASE_URL', URL, default='', example='https://api.anthropic.com/v1', render='commented', doc="Couche de compatibilité OpenAI d'Anthropic (selon Anthropic, pas une solution de\nproduction à long terme)."),
            S('ANTHROPIC_API_KEY', SECRET, default='', render='commented', scope=USER, doc='Clé du serveur ci-dessus (identifiant anthropic_api_key). Repli .env réservé aux admins ;\nun non-admin saisit sa propre clé dans Mon profil > Clés API (visible quand l\'adresse est déclarée).'),
        )),
        SubBlock('1.6', 'Google Gemini', settings=(
            S('GOOGLE_API_BASE_URL', URL, default='', example='https://generativelanguage.googleapis.com/v1beta/openai', render='commented', doc="Compatibilité OpenAI de l'API Gemini."),
            S('GOOGLE_API_KEY', SECRET, default='', render='commented', scope=USER, doc='Clé du serveur ci-dessus (identifiant google_api_key). Repli .env réservé aux admins ;\nun non-admin saisit sa propre clé dans Mon profil > Clés API (visible quand l\'adresse est déclarée).'),
        )),
        SubBlock('1.7', 'DeepSeek', settings=(
            S('DEEPSEEK_API_BASE_URL', URL, default='', example='https://api.deepseek.com', render='commented', doc='API DeepSeek, compatible OpenAI, sans /v1.'),
            S('DEEPSEEK_API_KEY', SECRET, default='', render='commented', scope=USER, doc='Clé du serveur ci-dessus (identifiant deepseek_api_key). Repli .env réservé aux admins ;\nun non-admin saisit sa propre clé dans Mon profil > Clés API (visible quand l\'adresse est déclarée).'),
        )),
        SubBlock('1.8', 'Qwen (Alibaba Cloud)', settings=(
            S('QWEN_API_BASE_URL', URL, default='', example='https://<WorkspaceId>.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1', render='commented', doc='Qwen, Alibaba Cloud Model Studio, mode compatible OpenAI : adresse propre à chaque\nespace de travail (Singapour ; États-Unis : https://dashscope-us.aliyuncs.com/compatible-mode/v1).'),
            S('QWEN_API_KEY', SECRET, default='', render='commented', scope=USER, doc='Clé du serveur ci-dessus (identifiant qwen_api_key). Repli .env réservé aux admins ;\nun non-admin saisit sa propre clé dans Mon profil > Clés API (visible quand l\'adresse est déclarée).'),
        )),
        SubBlock('1.9', 'GLM (Z.ai)', settings=(
            S('GLM_API_BASE_URL', URL, default='', example='https://api.z.ai/api/paas/v4', render='commented', doc='GLM (Z.ai, international), compatible OpenAI ; Chine : https://open.bigmodel.cn/api/paas/v4.'),
            S('GLM_API_KEY', SECRET, default='', render='commented', scope=USER, doc='Clé du serveur ci-dessus (identifiant glm_api_key). Repli .env réservé aux admins ;\nun non-admin saisit sa propre clé dans Mon profil > Clés API (visible quand l\'adresse est déclarée).'),
        )),
        SubBlock('1.10', 'Serveur local compatible OpenAI (Ollama, vLLM, LM Studio)', settings=(
            S('LOCAL_API_BASE_URL', URL, default='', example='http://localhost:11434/v1', render='commented', doc='Serveur local compatible OpenAI : Ollama http://localhost:11434/v1, vLLM\nhttp://localhost:8000/v1, LM Studio http://localhost:1234/v1.'),
            S('LOCAL_API_KEY', SECRET, default='', render='commented', doc="Clé du serveur local, s'il en exige une (Ollama : n'importe quelle valeur)."),
        )),
        SubBlock('1.11', 'Pinecone', settings=(
            S('PINECONE_API_KEY', SECRET, default='', example='pcsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', scope=USER, doc='Les index et collections doivent exister (pas de création automatique Pinecone) :\n3072 dimensions pour OpenAI, 1024 pour bge-m3 (Albert).'),
            S('PINECONE_ENV', STR, default='', scope=USER, replaced_by=(), doc="Facultatif : le client Pinecone v5 résout l'index par son nom ; champ conservé\npour le formulaire des identifiants."),
        )),
        SubBlock('1.12', 'Weaviate', settings=(
            S('WEAVIATE_URL', URL, default='', example='https://your-cluster.weaviate.cloud', scope=USER),
            S('WEAVIATE_API_KEY', SECRET, default='', example='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', scope=USER),
        )),
        SubBlock('1.13', 'Qdrant', settings=(
            S('QDRANT_URL', URL, default='', example='https://your-cluster.qdrant.io', scope=USER),
            S('QDRANT_API_KEY', SECRET, default='', example='XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', scope=USER),
        )),
        SubBlock('1.14', 'Zotero', settings=(
            S('ZOTERO_API_KEY', SECRET, default='', example='XXXXXXXXXXXXXXXXXXXXXXXX', scope=USER, doc='Clé : https://www.zotero.org/settings/keys/new (accès bibliothèque + notes,\nécriture pour pousser fiches et tags).'),
            S('ZOTERO_USER_ID', STR, default='', example='12345', scope=USER, doc='Bibliothèque personnelle (identifiant numérique affiché sur la page des clés).'),
            S('ZOTERO_GROUP_ID', STR, default='', scope=USER, doc='Bibliothèque de groupe, si les items y sont (sinon laisser vide).'),
        )),
        SubBlock('1.15', 'Resend (e-mails)', settings=(
            S('RESEND_API_KEY', SECRET, default='', example='re_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX', apply=RESTART, doc='Vérification des adresses et réinitialisation des mots de passe (Resend).'),
            S('RESEND_FROM_EMAIL', STR, default='onboarding@resend.dev', example='noreply@example.com', apply=RESTART, doc='Défaut : onboarding@resend.dev (adresse de test de Resend).'),
        )),
    )),
    Block('2', 'SERVEURS ET MODÈLES PAR SERVICE', doc="Chaque service déclare un couple SERVEUR (adresse d'un serveur du bloc 1) + MODÈLE\n(envoyé tel quel au serveur) ; vide = valeur par défaut (2.1), sauf pour l'OCR qui\nn'a pas de défaut. Embeddings, rerank et audio : lot 6.", subblocks=(
        SubBlock('2.1', 'Par défaut', settings=(
            S('LLM_DEFAULT_SERVER', SERVER, default='', example='https://openrouter.ai/api/v1', family='chat', doc="Serveur par défaut des services de langage : adresse d'un serveur du bloc 1.\nVide = mode historique (OPENROUTER_DEFAULT_MODEL, recodage gpt-4o-mini)."),
            S('LLM_DEFAULT_MODEL', MODEL, default='', example='google/gemini-3.8-flash', family='chat', doc='Modèle par défaut, envoyé tel quel au serveur retenu (nom du modèle chez ce serveur).'),
            S('OPENROUTER_DEFAULT_MODEL', MODEL, render='commented', default='gpt-4o-mini', example='google/gemini-3-flash-preview', scope=USER, family='chat', replaced_by=('LLM_DEFAULT_SERVER', 'LLM_DEFAULT_MODEL'), apply=RESTART, doc='Historique : lu seulement si LLM_DEFAULT_SERVER est vide. Modèle par défaut des\nnotes, fiches et citations (provider/model → OpenRouter, nom nu → OpenAI,\nalbert/<id> → Albert). Défaut : gpt-4o-mini.'),
        )),
        SubBlock('2.2', 'Recodage des chunks (étape 3.1)', settings=(
            S('LLM_RECODE_SERVER', SERVER, default='', example='', family='chat', doc='Serveur du recodage des chunks ; vide = LLM_DEFAULT_SERVER.'),
            S('LLM_RECODE_MODEL', MODEL, default='', example='openai/gpt-4o-mini', family='chat', doc='Modèle du recodage ; vide = LLM_DEFAULT_MODEL. Envoyé tel quel au serveur retenu.'),
            S('RECODE_MODEL', MODEL, render='commented', default='gpt-4o-mini-2024-07-18', family='chat', replaced_by=('LLM_RECODE_SERVER', 'LLM_RECODE_MODEL'), doc='Historique : snapshot daté qui remplace le modèle du recodage en mode durci,\nseulement si LLM_DEFAULT_SERVER est vide (sinon LLM_RECODE_MODEL fait foi).'),
            S('RECODE_PREFER_OPENAI', BOOL, render='commented', default='1', bool_style='01', replaced_by=('LLM_RECODE_SERVER',), doc='Historique (mode durci, LLM_DEFAULT_SERVER vide) : 1 = OpenAI direct (seed\nhonoré) ; 0 = OpenRouter (épinglage du backend requis pour cacher ce chemin).'),
        )),
        SubBlock('2.3', 'Notes Zotero (fiche et résumé)', settings=(
            S('LLM_NOTES_SERVER', SERVER, default='', example='', family='chat', doc='Serveur des notes Zotero (fiche, résumé) ; vide = LLM_DEFAULT_SERVER.'),
            S('LLM_NOTES_MODEL', MODEL, default='', example='', family='chat', doc='Modèle des notes ; vide = LLM_DEFAULT_MODEL.'),
        )),
        SubBlock('2.4', 'Fiches de lecture (structure, chapitres, synthèse)', settings=(
            S('LLM_BOOK_SERVER', SERVER, default='', example='', family='chat', doc='Serveur des fiches de lecture (structure, chapitres, synthèse) ; vide =\nLLM_DEFAULT_SERVER.'),
            S('LLM_BOOK_MODEL', MODEL, default='', example='', family='chat', doc='Modèle des fiches de lecture ; vide = LLM_DEFAULT_MODEL.'),
        )),
        SubBlock('2.5', 'Filtre et pré-filtre des citations', settings=(
            S('LLM_CITATIONS_SERVER', SERVER, default='', example='', family='chat', doc='Serveur du filtre et du pré-filtre des citations ; vide = LLM_DEFAULT_SERVER.'),
            S('LLM_CITATIONS_MODEL', MODEL, default='', example='', family='chat', doc='Modèle des citations ; vide = LLM_DEFAULT_MODEL.'),
        )),
        SubBlock('2.6', 'OCR', settings=(
            S('OCR_SERVER', SERVER, default='', example='https://api.mistral.ai/v1', family='ocr', doc="Serveur de l'OCR : adresse de Mistral, d'Albert ou du serveur local (bloc 1), ou le mot\nréservé local pour les moteurs internes. Un seul moteur, aucun repli : il doit\nrépondre au contrôle préalable, sinon rien ne démarre. OCR_MODEL vide = chaîne\nhistorique (Albert si OCR_ENABLE_ALBERT=1, Mistral, Docling, PyMuPDF)."),
            S('OCR_MODEL', MODEL, default='', example='mistral-ocr-latest', family='ocr', doc="Modèle d'OCR, envoyé tel quel : mistral-ocr-latest (Mistral), lightonocr-2-1b ou\nmistral-ocr-2512 (Albert ; /v1/ocr pour un modèle de type image-to-text, page par\npage sinon), docling ou pymupdf (couche texte native, sans Tesseract) avec local."),
            S('OCR_SERVER_FALLBACK', SERVER, default='', example='', family='ocr', doc="Repli de l'OCR : adresse d'un AUTRE serveur du bloc 1 (ou local). Vides tous les deux :\naucun repli, un échec du moteur principal fait échouer le document."),
            S('OCR_MODEL_FALLBACK', MODEL, default='', example='', family='ocr', doc="Modèle de repli, envoyé tel quel. Rempli : un document dont l'OCR principal échoue est\nrepris par ce moteur ; après une erreur de compte (clé, plafond, quota), la suite du lot\npasse par lui. Repli tracé (OCR_PROVIDER_FALLBACK) ; colonnes texteocr_server/model du\nmoteur qui a servi. Contrôlé avant le premier document, comme le moteur principal."),
            S('MISTRAL_OCR_MODEL', MODEL, render='commented', default='mistral-ocr-latest', scope=USER, family='ocr', replaced_by=('OCR_SERVER', 'OCR_MODEL'), doc='Historique (OCR_MODEL vide) : modèle OCR de Mistral.'),
            S('OCR_ENABLE_ALBERT', BOOL, render='commented', default='0', bool_style='01', replaced_by=('OCR_SERVER', 'OCR_MODEL'), doc='Historique (OCR_MODEL vide) : 1 = OCR Albert en tête de chaîne (exige ALBERT_ENABLED=1 et\nune clé) ; 0 (défaut) = chaîne OCR strictement identique.'),
            S('ALBERT_OCR_MODE', ENUM, render='commented', default='auto', choices=('auto', 'chat', 'ocr'), replaced_by=('OCR_MODEL',), doc='Historique (OCR_MODEL vide) : auto = /v1/ocr si le compte y a accès, sinon\nLightOnOCR ; chat = LightOnOCR seul ; ocr = /v1/ocr seul.'),
            S('ALBERT_OCR_CHAT_MODEL', MODEL, render='commented', default='lightonocr-2-1b', family='ocr', replaced_by=('OCR_MODEL',), doc='Historique (OCR_MODEL vide) : modèle OCR par chat sur pages rastérisées (id épinglé).'),
            S('ALBERT_OCR_DOC_MODEL', MODEL, render='commented', default='mistral-ocr-2512', family='ocr', replaced_by=('OCR_MODEL',), doc='Historique (OCR_MODEL vide) : modèle de /v1/ocr, accès restreint (id épinglé).'),
            S('OCR_ENABLE_OPENAI_FALLBACK', BOOL, render='commented', default='0', bool_style='01', replaced_by=('OCR_MODEL', 'OCR_SERVER_FALLBACK', 'OCR_MODEL_FALLBACK'), doc='Historique (OCR_MODEL vide) : OCR OpenAI Vision, plafonné à OPENAI_OCR_MAX_PAGES,\nil tronque les livres. Remplacé par OCR_SERVER_FALLBACK + OCR_MODEL_FALLBACK.'),
            S('OPENAI_OCR_MODEL', MODEL, render='commented', default='gpt-4o-mini', family='ocr', replaced_by=('OCR_MODEL', 'OCR_SERVER_FALLBACK', 'OCR_MODEL_FALLBACK'), doc="Historique (OCR_MODEL vide) : modèle de l'OCR OpenAI Vision."),
            S('OCR_ENABLE_LOCAL_FALLBACK', BOOL, render='commented', default='1', bool_style='01', replaced_by=('OCR_SERVER', 'OCR_MODEL'), doc="Historique (OCR_MODEL vide) : 1 = OCR local Docling avant le dernier recours PyMuPDF\n(sauté sans erreur si Docling n'est pas installé)."),
            S('LOCAL_OCR_ENGINE', ENUM, render='commented', default='docling', choices=('docling',), replaced_by=('OCR_MODEL',), doc="Historique (OCR_MODEL vide) : moteur de mise en page de l'OCR local (docling seul\nimplémenté). Avec OCR_MODEL : OCR_SERVER=local et OCR_MODEL=docling."),
        )),
        SubBlock('2.7', 'Embeddings', settings=(
            S('EMBEDDING_SERVER', SERVER, default='', example='', family='embedding', doc="Serveur des embeddings : adresse d'OpenAI, d'Albert ou d'un serveur compatible\n(Mistral, Google, Qwen, OpenRouter, serveur local) du bloc 1 ; vide = LLM_DEFAULT_SERVER.\nCouple vide : règle historique (EMBEDDING_PROVIDER)."),
            S('EMBEDDING_MODEL', MODEL, default='', example='', family='embedding', doc="Modèle d'embeddings, envoyé tel quel ; un modèle déclaré active le couple :\ntext-embedding-3-large (OpenAI, 3072 d), bge-m3 (Albert, 1024 d), mistral-embed…\nAutres serveurs : dimension mesurée au premier appel. Un index n'accepte qu'un\nespace (modèle et dimension) : changer de modèle exige un autre index."),
            S('EMBEDDING_PROVIDER', ENUM, render='commented', default='openai', choices=('openai', 'albert'), replaced_by=('EMBEDDING_SERVER', 'EMBEDDING_MODEL'), doc='Historique (EMBEDDING_MODEL vide) : openai (défaut : text-embedding-3-large, 3072 d)\nou albert (bge-m3, 1024 d, exige ALBERT_ENABLED=1). Deux espaces séparés.'),
            S('ALBERT_EMBED_MODEL', MODEL, render='commented', default='bge-m3', family='embedding', replaced_by=('EMBEDDING_MODEL',), doc="Historique (EMBEDDING_MODEL vide) : modèle d'embeddings Albert (id épinglé)."),
        )),
        SubBlock('2.8', 'Transcription audio', settings=(
            S('AUDIO_SERVER', SERVER, default='', example='', family='audio', doc="Serveur de la transcription : adresse d'Albert seulement (bloc 1)."),
            S('AUDIO_MODEL', MODEL, default='', example='', family='audio', doc="Modèle de transcription, envoyé tel quel (whisper-large-v3) ; un modèle déclaré\nactive l'import d'enregistrements (mp3, wav ; autres formats convertis par ffmpeg),\npuis pipeline habituel (recodage sauté). Vide : règle historique."),
            S('ALBERT_AUDIO_ENABLED', BOOL, render='commented', default='0', bool_style='01', replaced_by=('AUDIO_MODEL',), doc="Historique (AUDIO_MODEL vide) : 1 = import d'enregistrements transcrits par Whisper."),
            S('ALBERT_AUDIO_MODEL', MODEL, render='commented', default='whisper-large-v3', family='audio', replaced_by=('AUDIO_SERVER', 'AUDIO_MODEL'), doc='Historique (AUDIO_MODEL vide) : modèle de transcription Albert.'),
        )),
        SubBlock('2.9', 'Interrupteur Albert, politique et repli des modèles', settings=(
            S('ALBERT_ENABLED', BOOL, default='0', bool_style='01', doc="Interrupteur maître des services Albert (DINUM) : chat, OCR, embeddings bge-m3,\ncollections. 0 (défaut) = comportement strictement identique : ni appel, ni route,\nni interface, ni clé JSON Albert. Albert désactivé, deux réglages s'appliquent\nquand même : EMBEDDING_PROVIDER=albert est refusé (phase dense en exit 2, champ du\nformulaire en 400) et DEDUP_SIM_THRESHOLD_BGE_M3 s'applique au Tier 3 de tout\nfichier bge-m3. Guide complet : .claude/docs/albert.md."),
            S('ALBERT_DATA_POLICY', ENUM, default='compatible', choices=('compatible', 'albert_only'), doc="compatible (défaut) = fournisseurs historiques conservés à côté d'Albert ;\nalbert_only = inférence Albert et traitements locaux seulement (OpenAI,\nOpenRouter et Mistral direct refusés, modèle vide = défaut Albert du rôle).\nalbert_only avec ALBERT_ENABLED=0 : refus explicite, jamais de retour silencieux."),
            S('ALBERT_MODEL_FALLBACK', BOOL, default='1', bool_style='01', doc='1 = repli sur le modèle suivant du rôle après des 503 répétés (jamais sur 404,\njamais pour les embeddings).'),
        )),
    )),
    Block('3', 'OCR : RÉGLAGES PAR MOTEUR', doc='', subblocks=(
        SubBlock('3.1', 'Commun', settings=(
            S('PDF_EXTRACTION_WORKERS', INT, default='1', example='3', doc='PDF OCRisés en parallèle (défaut : 1 = séquentiel) ; aligner sur\nMISTRAL_CONCURRENT_CALLS.'),
            S('OCR_MIN_CHARS_PER_PAGE', INT, default='500', doc="Garde-fou anti-troncature : en dessous de cette moyenne de caractères par page,\nun résultat OpenAI ou legacy est marqué partiel (colonne texteocr_partial,\nOCR_PARTIAL dans *_errors.json) au lieu d'un succès silencieux."),
        )),
        SubBlock('3.2', 'Mistral', settings=(
            S('MISTRAL_CONCURRENT_CALLS', INT, default='3', doc="Appels OCR Mistral simultanés par processus (sémaphore). Aligner\nPDF_EXTRACTION_WORKERS sur cette valeur : plus de workers que de créneaux\nne fait qu'attendre et multiplie les 429."),
            S('MISTRAL_OCR_TIMEOUT', INT, default='300', doc="Délai (s) d'une requête HTTP d'OCR Mistral."),
            S('MISTRAL_DELETE_UPLOADED_FILE', BOOL, default='1', bool_style='01', doc="1 = supprime le fichier téléversé chez Mistral une fois l'OCR terminé."),
            S('MISTRAL_MAX_UPLOAD_MB', FLOAT, default='45', doc="Gros fichiers : /v1/files refuse ~50 Mo et l'OCR refuse plus de 1000 pages\n(code 3730). Au-delà de MISTRAL_MAX_UPLOAD_MB, le PDF est recompressé\n(PyMuPDF) ; s'il reste trop gros, ou au-delà de MISTRAL_MAX_PAGES, il est\ndécoupé en parts de MISTRAL_SPLIT_PART_MB et MISTRAL_SPLIT_PART_PAGES\n(les deux critères), OCRisées une à une puis recollées avec une\nnumérotation de pages globale."),
            S('MISTRAL_AUTO_COMPRESS', BOOL, default='true', bool_style='truefalse'),
            S('MISTRAL_AUTO_SPLIT', BOOL, default='true', bool_style='truefalse'),
            S('MISTRAL_SPLIT_PART_MB', FLOAT, default='30'),
            S('MISTRAL_MAX_PAGES', INT, default='950'),
            S('MISTRAL_SPLIT_PART_PAGES', INT, default='500'),
            S('MISTRAL_OCR_RETRIES', INT, default='4', doc="Réessais des erreurs transitoires (404 « Could not get file », 408, 429,\n5xx, coupure réseau) avant de passer au maillon suivant. Un 401 n'est\njamais réessayé. Backoff exponentiel (BACKOFF * 2**tentative) plafonné à\nMAX_BACKOFF, avec gigue ; l'en-tête Retry-After d'un 429 est respecté."),
            S('MISTRAL_OCR_RETRY_BACKOFF', FLOAT, default='3.0'),
            S('MISTRAL_OCR_RETRY_MAX_BACKOFF', FLOAT, default='60.0'),
            S('MISTRAL_OCR_RATE_LIMIT_BACKOFF', FLOAT, default='10.0', doc='429 (limite de débit) : échelle plus longue (10, 20, 40, 80 s, plafond\n120 s) et pause commune à tous les threads du processus, y compris après\nla dernière tentative (la part ou le document suivant attend aussi).'),
            S('MISTRAL_OCR_RATE_LIMIT_MAX_BACKOFF', FLOAT, default='120.0'),
            S('MISTRAL_SPLIT_RETRY_PASSES', INT, default='1', doc="Livre découpé : une part épuisée sur un 429 est reprise après les autres\n(RETRY_PASSES passes supplémentaires, chacune après RETRY_DELAY s ; 0 =\naucune reprise). Au-delà de MAX_RATE_LIMITED_PARTS parts consécutives en\n429, la passe n'envoie plus rien (quota épuisé) et remet le reste à la\npasse suivante (0 = jamais). Une part encore en échec laisse un marqueur\n« OCR ÉCHOUÉ » et le document est signalé partiel (cause : 429)."),
            S('MISTRAL_SPLIT_RETRY_DELAY', FLOAT, default='60.0'),
            S('MISTRAL_SPLIT_MAX_RATE_LIMITED_PARTS', INT, default='2'),
        )),
        SubBlock('3.3', 'Albert', settings=(
            S('ALBERT_OCR_DPI', INT, default='200', doc='Rastérisation et requête LightOnOCR : résolution, plus grand côté (pixels),\nmax_tokens, température, top_p.'),
            S('ALBERT_OCR_MAX_SIDE', INT, default='1540'),
            S('ALBERT_OCR_MAX_TOKENS', INT, default='4096'),
            S('ALBERT_OCR_TEMPERATURE', FLOAT, default='0.2'),
            S('ALBERT_OCR_TOP_P', FLOAT, default='0.9'),
            S('ALBERT_OCR_MAX_PAGES', INT, default='0', doc='Plafond de pages par document (0 = aucun ; au-delà, résultat partiel).'),
            S('ALBERT_OCR_MAX_FAILED_RATIO', FLOAT, default='0.5', doc='Part de pages en échec au-delà de laquelle le maillon Albert est abandonné.'),
            S('ALBERT_OCR_PART_MB', FLOAT, default='15', doc='Découpage des envois à /v1/ocr : taille (Mo) et nombre de pages par part.'),
            S('ALBERT_OCR_PART_PAGES', INT, default='100'),
            S('ALBERT_OCR_CONCURRENCY', INT, default='1', doc='Requêtes OCR Albert simultanées par processus.'),
            S('ALBERT_OCR_SKIP_RECODE', BOOL, default='0', bool_style='01', doc='1 = texte albert_lightonocr exclu du recodage. 0 (défaut prudent) = recodé.'),
            S('ALBERT_OCR_CHECKPOINT', BOOL, default='1', bool_style='01', doc="1 = reprise page par page de l'OCR LightOnOCR (pages validées gardées dans le\ndossier de sortie de la session, jamais renvoyées à Albert lors d'une relance)."),
        )),
        SubBlock('3.4', 'Local (Docling)', settings=(
            S('LOCAL_OCR_PYTHON', PATH, default='', render='commented', locked="chemin d'un programme exécuté par le serveur", default_rule="/opt/ocr-venv/bin/python s'il existe, sinon l'interpréteur courant", doc='Interpréteur du sous-processus OCR (défaut : /opt/ocr-venv/bin/python).'),
            S('LOCAL_OCR_ENGINE_OCR', ENUM, default='tesseract', choices=('tesseract', 'rapidocr'), doc='Moteur de reconnaissance de Docling : tesseract (FR/EN, hors ligne) ou\nrapidocr (onnxruntime et modèles à installer à part ; plus lent sur CPU ARM).'),
            S('LOCAL_OCR_USE_CLS', BOOL, default='0', bool_style='01', doc="1 = classifieur d'orientation de RapidOCR (inutile sur des scans droits)."),
            S('LOCAL_OCR_DEVICE', ENUM, default='cpu', choices=('cpu', 'cuda'), doc='cpu | cuda'),
            S('LOCAL_OCR_MAX_PAGES', INT, default='0', doc='Pages par document (0 = aucun plafond).'),
            S('LOCAL_OCR_TIMEOUT', INT, default='1800', doc='Délai (s) par document.'),
            S('LOCAL_OCR_CONCURRENCY', INT, default='1', doc='Documents OCRisés en parallèle (CPU et RAM : garder 1 ou 2).'),
            S('LOCAL_OCR_THREADS', INT, default='0', doc='Threads torch et mise en page (0 = min(nombre de CPU, 8)).'),
            S('LOCAL_OCR_TABLES', BOOL, default='0', bool_style='01', doc='1 = détection des tableaux (TableFormer, nettement plus lent).'),
            S('LOCAL_OCR_IMAGE_SCALE', FLOAT, default='0', doc='Échelle de rendu des pages (1.0 ≈ 72 DPI ; 0 = défaut de Docling).'),
        )),
        SubBlock('3.5', 'OpenAI Vision (désactivé, retiré au lot 5)', settings=(
            S('OPENAI_OCR_MAX_PAGES', INT, render='commented', default='10', replaced_by=('OCR_MODEL', 'OCR_SERVER_FALLBACK', 'OCR_MODEL_FALLBACK'), doc="Réglages de l'OCR OpenAI Vision, utilisés seulement avec OCR_ENABLE_OPENAI_FALLBACK=1."),
            S('OPENAI_OCR_MAX_TOKENS', INT, render='commented', default='2048', replaced_by=('OCR_MODEL', 'OCR_SERVER_FALLBACK', 'OCR_MODEL_FALLBACK')),
            S('OPENAI_OCR_RENDER_SCALE', FLOAT, render='commented', default='2.0', replaced_by=('OCR_MODEL', 'OCR_SERVER_FALLBACK', 'OCR_MODEL_FALLBACK'), doc='Facteur de rendu des pages envoyées au modèle de vision.'),
            S('OPENAI_OCR_PROMPT', STR, default='Transcris cette page PDF en Markdown lisible sans résumer ni modifier le contenu.', render='commented', replaced_by=(), doc='Consigne envoyée pour chaque page (défaut : transcription Markdown fidèle).'),
        )),
    )),
    Block('4', 'DÉCOUPAGE, RECODAGE ET EMBEDDINGS', doc='', subblocks=(
        SubBlock('4.1', 'Découpage et parallélisme', settings=(
            S('DEFAULT_MAX_WORKERS', INT, default='', example='8', default_rule='nombre de CPU − 1 (au moins 1)', doc="Threads du chunking et des appels d'API (défaut : nombre de CPU − 1).\nNe jamais laisser vide (la page /health/detailed lit un entier)."),
            S('DEFAULT_DOC_WORKERS', INT, default='3', example='6', doc='Documents traités en parallèle au chunking (défaut : 3).'),
            S('CHUNK_CHECKPOINT_SECONDS', FLOAT, default='30', doc='Délai minimal (s) entre deux écritures atomiques du fichier de chunks.'),
        )),
        SubBlock('4.2', 'Recodage', settings=(
            S('DEFAULT_BATCH_SIZE_GPT', INT, default='5', doc='Chunks par lot de recodage GPT.'),
            S('RECODE_HARDEN_ENABLED', BOOL, default='0', bool_style='01', doc="Durcissement du décodage (température 0, top_p 1, seed, snapshot daté),\nindépendant du cache. Avec 1, RECODE_MODEL remplace le modèle choisi (jusqu'au lot 4)."),
            S('RECODE_TEMPERATURE', FLOAT, default='0.0', doc='Paramètres du décodage durci.'),
            S('RECODE_TOP_P', FLOAT, default='1.0'),
            S('RECODE_SEED', INT, default='0'),
            S('RECODE_MAX_TOKENS', INT, default='1600', doc='Borné ; sûr seulement avec la garde finish_reason (fallback_truncated).'),
            S('RECODE_OPENROUTER_PROVIDER', STR, default='', render='commented', doc='Backend OpenRouter épinglé (allow_fallbacks=False), obligatoire pour cacher\nle chemin OpenRouter.'),
        )),
        SubBlock('4.3', 'Caches', settings=(
            S('RECODE_CACHE_ENABLED', BOOL, default='0', bool_style='01', doc="Cache SQLite du recodage indexé par content_hash : un succès déjà vu coûte\n0 appel LLM et redonne un texte identique à l'octet. 0 (défaut) = inchangé."),
            S('RECODE_EMBED_CACHE_ENABLED', BOOL, default='0', bool_style='01', doc='Cache des vecteurs denses (texte identique → vecteur gelé).'),
            S('RECODE_CACHE_PATH', PATH, default='data/recode_cache.sqlite', locked="chemin d'écriture du serveur"),
        )),
        SubBlock('4.4', 'Embeddings', settings=(
            S('DEFAULT_EMBEDDING_BATCH_SIZE', INT, default='32', doc="Chunks par lot d'embeddings OpenAI."),
            S('EMBEDDING_FUTURES_PER_WORKER', INT, default='4', doc="Lots d'embeddings en vol par worker (fenêtre bornée)."),
            S('EMBED_MAX_MISSING_RATIO', FLOAT, default='0', doc="Part d'embeddings OpenAI manquants tolérée avant l'échec de la phase dense\n(0 à 1 ; 0 = aucun manque, la phase sort en erreur)."),
            S('ALBERT_EMBED_BATCH', INT, default='64', doc="Textes par requête d'embeddings (plafonné à 64)."),
            S('ALBERT_EMBED_MAX_MISSING_RATIO', FLOAT, default='0.0', doc="Part d'embeddings manquants tolérée avant l'échec de la phase (0.0 = aucun manque)."),
            S('ALBERT_EMBED_L2_NORMALIZE', BOOL, default='1', bool_style='01', doc="1 = vecteurs normalisés L2 (norme enregistrée dans l'espace vectoriel)."),
        )),
    )),
    Block('5', 'NOTES, FICHES, CITATIONS ET AUDIO', doc='', subblocks=(
        SubBlock('5.1', 'Notes et fiches', settings=(
            S('MAX_CONCURRENT_LLM_CALLS', INT, default='5', apply=RESTART, doc='Appels LLM simultanés sur toute la plateforme (notes, fiches, citations ;\nsémaphore global partagé entre utilisateurs).'),
            S('BOOK_NOTE_VERSION', ENUM, default='v2', choices=('v1', 'v2'), apply=RESTART, doc='Notes de livres en plusieurs phases : v2 (contrôles qualité après\nassemblage) ou v1 (sans ces contrôles).'),
        )),
        SubBlock('5.2', 'Citations (Publish or Perish)', settings=(
            S('MAX_CONCURRENT_WEB_FETCHES', INT, default='10', apply=RESTART, doc='Import Publish or Perish : pages web récupérées en parallèle et citations par\nlot (1 à 50).'),
            S('DEFAULT_CITATION_BATCH_SIZE', INT, default='10', apply=RESTART),
        )),
        SubBlock('5.3', 'Audio (Albert)', settings=(
            S('ALBERT_AUDIO_RPM', INT, default='45', doc='Quota et concurrence de la transcription.'),
            S('ALBERT_AUDIO_CONCURRENCY', INT, default='1'),
            S('ALBERT_AUDIO_LANGUAGE', STR, default='fr', doc='Langue (code ISO 639-1 ; auto = détection par le modèle).'),
            S('ALBERT_AUDIO_SEGMENT_SECONDS', INT, default='600', doc='Durée des segments découpés par ffmpeg pour les fichiers longs (30 à 3600 s).'),
            S('ALBERT_TIMEOUT_AUDIO', FLOAT, default='300', doc="Délai HTTP (s) d'une requête de transcription."),
        )),
    )),
    Block('6', 'DÉDUPLICATION ET BASES VECTORIELLES', doc='', subblocks=(
        SubBlock('6.1', 'Déduplication des chunks', settings=(
            S('DEDUP_ENABLED', BOOL, default='0', bool_style='01', doc='Interrupteur maître. 0 (défaut) = comportement strictement identique : aucun\ncontent_hash écrit dans le JSON, id aléatoire conservé, dedup_filter inactif.'),
            S('DEDUP_SEMANTIC', BOOL, default='0', bool_style='01', doc="Tier 3 sémantique (similarité d'embedding sur les survivants du Tier 2).\nRéservé aux petits lots ré-OCRisés par plusieurs fournisseurs."),
            S('DEDUP_SIM_THRESHOLD', FLOAT, default='0.97', doc='Seuil du Tier 3 (les métadonnées et le Jaccard portent la décision).'),
            S('DEDUP_TEXT_JACCARD', FLOAT, default='0.92', doc='Jaccard minimal sur les trigrammes de mots (Tier 3).'),
            S('DEDUP_META_FIELDS', LIST, default='title', doc="Champs qui identifient la SOURCE d'un chunk (id v2 = contenu + source, audit\nA05) et doivent concorder avant un refus (ex. title,authors). Titres\ngénériques (« Introduction ») : ajouter authors ou itemKey, sinon deux œuvres\nhomonymes au passage identique partagent un id. Champ vide = jamais dédupliqué."),
            S('DEDUP_MIN_CHARS', INT, default='64', doc="Plancher de texte normalisé ; en deçà, le chunk n'est jamais comparé."),
            S('DEDUP_QUERY_BATCH', INT, default='100', doc="Taille du lot d'existence (1 aller-retour par lot, pas par chunk)."),
            S('DEDUP_MATCH_MAX', INT, default='3', doc='Témoins existants listés au plus par refus.'),
            S('DEDUP_SEMANTIC_MAX_SURVIVORS', INT, default='500', doc='Garde : au-delà de ce nombre de survivants, le Tier 3 est sauté.'),
            S('DEDUP_SIM_THRESHOLD_BGE_M3', FLOAT, default='', doc='Seuil Tier 3 de la dédup pour bge-m3 ; vide (défaut) = Tier 3 sauté pour bge-m3.'),
            S('DEDUP_JOURNAL_DIR', PATH, default='', render='commented', locked="chemin d'écriture du serveur", default_rule="à côté du JSON d'embeddings", doc="Dossier du journal dedup_journal.jsonl (défaut : à côté du JSON d'embeddings)."),
        )),
        SubBlock('6.2', 'Envoi vers Pinecone, Weaviate, Qdrant', settings=(
            S('PINECONE_BATCH_SIZE', INT, default='100', doc="Taille des lots d'insertion."),
            S('WEAVIATE_BATCH_SIZE', INT, default='100'),
            S('QDRANT_BATCH_SIZE', INT, default='100'),
        )),
        SubBlock('6.3', 'Collections Albert', settings=(
            S('ALBERT_METADATA_FIELDS', LIST, default='content_id,content_hash,chunk_index,total_chunks,title,authors,year,doi,item_key,filename', doc='Métadonnées envoyées aux collections : 10 au plus, avec content_id, content_hash,\nchunk_index et DEDUP_META_FIELDS ; jamais path.'),
            S('ALBERT_TIMEOUT_COLLECTIONS', FLOAT, default='120', doc='Délai HTTP (s) des appels aux collections.'),
        )),
    )),
    Block('7', 'ALBERT : QUOTAS, CONCURRENCE, RÉESSAIS, DÉLAIS ET TRAÇABILITÉ', doc='', subblocks=(
        SubBlock('7.1', 'Quotas (partagés par compte)', settings=(
            S('ALBERT_RECODE_RPM', INT, default='45', doc="Limiteur proactif, environ 90 % des quotas du régime d'expérimentation : requêtes\npar minute des rôles recodage, notes, OCR et embeddings, puis tokens d'entrée\nestimés par minute pour les rôles de chat."),
            S('ALBERT_NOTES_RPM', INT, default='9'),
            S('ALBERT_OCR_RPM', INT, default='45'),
            S('ALBERT_EMBED_RPM', INT, default='450'),
            S('ALBERT_CHAT_TPM', INT, default='115000'),
            S('ALBERT_PROCESS_SHARE', FLOAT, default='1.0', doc='Part des quotas attribuée à ce processus (0 < part <= 1).'),
            S('ALBERT_TOKEN_ESTIMATOR', ENUM, default='chars', choices=('chars', 'tiktoken'), doc="Estimation des tokens d'entrée du limiteur (chars = longueur / 3 ; ou tiktoken)."),
        )),
        SubBlock('7.2', 'Limiteur', settings=(
            S('ALBERT_LIMITER_BACKEND', ENUM, render='commented', default='local', choices=('local', 'redis'), doc="local = limiteur par processus ; redis = partagé (recommandé avec Celery ou\nplusieurs sessions). Laisser en commentaire sous Docker Compose : le défaut y est\nredis (docker-compose.yml), et une valeur écrite ici le remplace. Hors Docker, le\ndéfaut du code est local."),
            S('ALBERT_LIMITER_REDIS_URL', URL, default='', locked="un Redis étranger recevrait l'état du limiteur", default_rule='CELERY_BROKER_URL, sinon redis://localhost:6379/0', doc='URL Redis du limiteur ; vide = CELERY_BROKER_URL, sinon redis://localhost:6379/0.'),
            S('ALBERT_LIMITER_REQUIRE_REDIS', BOOL, default='0', bool_style='01', doc='1 = avec ALBERT_LIMITER_BACKEND=redis, aucun nouvel appel si Redis est injoignable\n(au lieu du seau local par processus, qui ne garantit pas le plafond du compte).'),
        )),
        SubBlock('7.3', 'Concurrence', settings=(
            S('ALBERT_RECODE_CONCURRENCY', INT, default='2', doc='Appels Albert simultanés par processus : recodage, notes et fiches, embeddings,\nenvois de chunks vers une collection.'),
            S('ALBERT_NOTES_CONCURRENCY', INT, default='2'),
            S('ALBERT_EMBED_CONCURRENCY', INT, default='4'),
            S('ALBERT_PUSH_CONCURRENCY', INT, default='1'),
        )),
        SubBlock('7.4', 'Raisonnement', settings=(
            S('ALBERT_REASONING_EFFORT', ENUM, default='medium', choices=('low', 'medium', 'high'), doc='Effort de raisonnement des modèles gpt-oss (low, medium, high).'),
            S('ALBERT_REASONING_HEADROOM', INT, default='2048', doc='Tokens ajoutés à max_tokens pour le raisonnement des modèles gpt-oss.'),
        )),
        SubBlock('7.5', 'Réessais et délais', settings=(
            S('ALBERT_BUSY_RETRIES', INT, default='2', doc='Essais courts sur 503 avant le repli de rôle.'),
            S('ALBERT_MAX_RETRIES', INT, default='4', doc='Réessais au plus sur erreur transitoire ; backoff exponentiel (base et plafond,\nen secondes).'),
            S('ALBERT_RETRY_BACKOFF', FLOAT, default='2.0'),
            S('ALBERT_RETRY_MAX_BACKOFF', FLOAT, default='60'),
            S('ALBERT_RETRY_AFTER_MAX', FLOAT, default='120', doc='Retry-After (secondes) au-delà duquel un 429 signifie un quota épuisé (abandon).'),
            S('ALBERT_TIMEOUT_CHAT', FLOAT, default='120', doc='Délais HTTP (secondes) : chat, notes et fiches, page OCR, part /v1/ocr,\nembeddings.'),
            S('ALBERT_TIMEOUT_NOTES', FLOAT, default='300'),
            S('ALBERT_TIMEOUT_OCR_PAGE', FLOAT, default='120'),
            S('ALBERT_TIMEOUT_OCR_DOC', FLOAT, default='300'),
            S('ALBERT_TIMEOUT_EMBED', FLOAT, default='60'),
            S('ALBERT_SUBPROCESS_TIMEOUT', INT, default='21600', doc='Délai (secondes) des sous-processus quand la requête sélectionne Albert (6 h) ;\nsinon, délais historiques inchangés.'),
        )),
        SubBlock('7.6', 'Traçabilité', settings=(
            S('ALBERT_USAGE_LOG', BOOL, default='1', bool_style='01', doc='1 = ledger albert_usage.jsonl écrit quand Albert a été appelé.'),
            S('ALBERT_PREFLIGHT', BOOL, default='1', bool_style='01', doc='1 = contrôle du compte et des modèles (/v1/me, /v1/models) avant un traitement.'),
        )),
    )),
    Block('8', 'TRAITEMENTS : ADMISSION, IMPORTS ET SESSIONS', doc='', subblocks=(
        SubBlock('8.1', 'Admission (audit A12)', settings=(
            S('MAX_ACTIVE_JOBS', INT, default='8', doc='Traitements actifs sur le serveur et par utilisateur (429 au-delà ; 0 = sans\nlimite). Un traitement du même groupe sur une session déjà occupée est\nrefusé (409). Verrous partagés entre processus et conteneurs dans\nRAGPY_LOCK_DIR (vide = data/locks).'),
            S('MAX_ACTIVE_JOBS_PER_USER', INT, default='3'),
            S('RAGPY_LOCK_DIR', PATH, default='', locked="chemin d'écriture du serveur", default_rule='<dépôt>/data/locks'),
        )),
        SubBlock('8.2', 'Imports (audit A01)', settings=(
            S('UPLOAD_MAX_MB', INT, default='4096', doc="Limites (Mo ; 0 = pas de limite) du fichier envoyé et de l'archive ZIP\ndécompressée, puis nombre de membres d'une archive (413 au-delà). Aligner\nclient_max_body_size dans nginx."),
            S('UPLOAD_MAX_UNZIPPED_MB', INT, default='16384'),
            S('UPLOAD_MAX_ZIP_ENTRIES', INT, default='100000'),
        )),
        SubBlock('8.3', 'Sessions et nettoyage', settings=(
            S('SESSION_TTL_HOURS', INT, default='24', doc="Durée de vie (heures) d'une session de traitement avant suppression de son\ndossier uploads/<session>."),
            S('CLEANUP_ENABLED', BOOL, default='true', bool_style='truefalse', apply=RESTART, doc="Nettoyage périodique des sessions expirées (planificateur de l'application\nweb, pas Celery Beat)."),
            S('CLEANUP_INTERVAL_HOURS', INT, default='6', apply=RESTART),
        )),
    )),
    Block('9', 'COMPTES, SÉCURITÉ ET DÉPLOIEMENT', doc='', subblocks=(
        SubBlock('9.1', 'Environnement et secret JWT (audit A09)', settings=(
            S('RAGPY_ENV', ENUM, default='development', scope=DEPLOY, choices=('development', 'production'), locked='désactiverait le contrôle de production', apply=RESTART, doc='production = démarrage refusé si JWT_SECRET_KEY est absent, vaut la valeur de\nremplacement du dépôt ou fait moins de 32 caractères (development : alerte\nseule dans les journaux).'),
            S('JWT_SECRET_KEY', SECRET, default='', scope=DEPLOY, locked="secret de chiffrement des identifiants : rotation par scripts/rotate_credentials_key.py, jamais depuis l'interface", default_rule='valeur de remplacement publique (refusée en production)', apply=RESTART, doc='Secret JWT, défini une fois : il dérive aussi la clé Fernet des identifiants\nstockés. Vide = valeur de remplacement publique (refusée en production).\nGénérer : python -c "import secrets; print(secrets.token_urlsafe(48))"'),
            S('JWT_SECRET_KEY_PREVIOUS', SECRET, default='', scope=DEPLOY, locked="secret de chiffrement des identifiants : rotation par scripts/rotate_credentials_key.py, jamais depuis l'interface", apply=RESTART, doc="Rotation : l'ancien secret ici le temps de rechiffrer\n(scripts/rotate_credentials_key.py), puis vider. Sans lui, changer\nJWT_SECRET_KEY rend illisibles les clés enregistrées par les utilisateurs."),
            S('JWT_ALGORITHM', ENUM, default='HS256', scope=DEPLOY, choices=('HS256', 'HS384', 'HS512'), locked='sécurité des sessions', apply=RESTART),
            S('JWT_ACCESS_TOKEN_EXPIRE_MINUTES', INT, default='30', scope=DEPLOY, apply=RESTART, doc="Durée de vie des jetons d'accès (minutes) et de rafraîchissement (jours)."),
            S('JWT_REFRESH_TOKEN_EXPIRE_DAYS', INT, default='7', scope=DEPLOY, apply=RESTART),
        )),
        SubBlock('9.2', 'Accès web et base de données', settings=(
            S('CORS_ORIGINS', LIST, default='http://localhost:8000,http://127.0.0.1:8000', scope=DEPLOY, locked='accès inter-origines avec cookies', apply=RESTART, doc="Origines cross-origin autorisées (avec cookies), séparées par des virgules ;\nles pages sont servies par l'application elle-même."),
            S('APP_URL', URL, default='http://localhost:8000', scope=DEPLOY, apply=RESTART, doc="URL publique de l'application (liens des e-mails de vérification et de\nréinitialisation). Ex. https://ragpy.example.org derrière nginx."),
            S('DATABASE_URL', STR, default='', example='sqlite:////app/data/ragpy.db', render='commented', scope=DEPLOY, locked='base de données du serveur', default_rule='sqlite:///<dépôt>/data/ragpy.db', apply=RESTART, doc='Base SQLite par défaut : data/ragpy.db du dépôt (/app/data sous Docker).\nNe jamais laisser la ligne vide : la commenter pour garder le défaut.'),
            S('DEBUG', BOOL, default='false', scope=DEPLOY, bool_style='truefalse', locked='désactive le cookie sécurisé et expose des jetons de réinitialisation', apply=RESTART, doc='true = requêtes SQL tracées dans les journaux.'),
        )),
        SubBlock('9.3', 'Comptes et e-mails', settings=(
            S('EMAIL_VERIFICATION_EXPIRE_HOURS', INT, default='24', apply=RESTART, doc='Validité (heures) des liens de vérification et de réinitialisation.'),
            S('PASSWORD_RESET_EXPIRE_HOURS', INT, default='1', apply=RESTART),
            S('USERS_SANDBOX', BOOL, default='FALSE', example='TRUE', bool_style='TRUEFALSE', apply=RESTART, doc="TRUE = toute nouvelle inscription non-admin attend l'approbation d'un\nadministrateur (Admin > Utilisateurs). Défaut : FALSE."),
        )),
    )),
    Block('10', 'SERVEUR, CELERY, REDIS, FLOWER ET IMAGE DOCKER', doc='', subblocks=(
        SubBlock('10.1', 'Serveur Uvicorn (image Docker)', settings=(
            S('UVICORN_WORKERS', INT, default='1', scope=DEPLOY, apply=RECREATE, locked="démarrage de l'image (garder 1 worker)", doc="Lu par la commande de démarrage de l'image (en local, passer les options à\nuvicorn). Garder 1 worker : le registre des scripts en cours (bouton Arrêt),\nles flux SSE et le planificateur de nettoyage vivent dans le processus ;\nplusieurs workers les dispersent."),
            S('UVICORN_TIMEOUT_KEEP_ALIVE', INT, default='120', scope=DEPLOY, apply=RECREATE, locked="démarrage de l'image", doc='Keep-alive (s), pour les traitements longs.'),
            S('UVICORN_LIMIT_CONCURRENCY', INT, default='100', scope=DEPLOY, apply=RECREATE, locked="démarrage de l'image", doc='Requêtes simultanées au plus par worker.'),
        )),
        SubBlock('10.2', 'Celery et Redis (mode file de tâches)', settings=(
            S('ENABLE_CELERY', BOOL, default='false', scope=DEPLOY, bool_style='truefalse', apply=RESTART, doc='true = endpoints /api/celery/* actifs (sinon : traitements en sous-processus\navec progression SSE, mode par défaut).'),
            S('CELERY_BROKER_URL', URL, default='redis://localhost:6379/0', scope=DEPLOY, locked='un broker étranger pourrait injecter des tâches', apply=RESTART, doc='Broker et résultats. Sous Docker Compose, forcés à redis://redis:6379/0\n(réseau interne) quelle que soit la valeur ci-dessous.'),
            S('CELERY_RESULT_BACKEND', URL, default='redis://localhost:6379/0', scope=DEPLOY, locked='un broker étranger pourrait injecter des tâches', apply=RESTART),
            S('REDIS_BIND', STR, default='127.0.0.1', scope=DEPLOY, apply=RECREATE, locked='exposition réseau de Redis', doc="Adresse de publication de Redis sur l'hôte (sans mot de passe) : 127.0.0.1 par\ndéfaut ; 0.0.0.0 seulement derrière un pare-feu."),
            S('ORPHAN_PROCESS_MAX_RUNTIME', INT, default='', default_rule="plus grande limite de temps d'une tâche (Albert compris) + 600 s", doc="Âge (s) au-delà duquel un script du worker est tué comme orphelin ; vide =\nplus grande limite de temps d'une tâche (Albert compris) + 600 s."),
        )),
        SubBlock('10.3', 'Flower', settings=(
            S('FLOWER_BIND', STR, default='127.0.0.1', scope=DEPLOY, apply=RECREATE, locked='exposition réseau de Flower', doc="Adresse de publication de Flower sur l'hôte : 127.0.0.1 par défaut ; 0.0.0.0\nseulement derrière un pare-feu."),
            S('FLOWER_USER', STR, default='', scope=DEPLOY, apply=RECREATE, locked='accès à la supervision', doc='Obligatoires : Flower refuse de démarrer sans identifiants ou avec admin/admin,\nadmin/changeme (audit A09). Accès : tunnel SSH vers le port 5555.'),
            S('FLOWER_PASSWORD', SECRET, default='', scope=DEPLOY, apply=RECREATE, locked='accès à la supervision'),
        )),
        SubBlock('10.4', "Construction de l'image", settings=(
            S('INSTALL_LOCAL_OCR', BOOL, default='false', scope=BUILD, apply=REBUILD, bool_style='truefalse', locked="lu à la construction de l'image", doc="true = Docling (OCR local) installé dans l'image Docker ; lu à la construction\n(docker compose up -d --build)."),
            S('INSTALL_FFMPEG', BOOL, default='false', scope=BUILD, apply=REBUILD, bool_style='truefalse', locked="lu à la construction de l'image", doc="Audio Albert (sprint R2) : true = ffmpeg dans l'image Docker (conversion et\ndécoupage des enregistrements longs) ; lu à la construction (docker compose up -d --build)."),
        )),
    )),
    Block('11', 'SUPERVISION (PROMETHEUS, FACULTATIF)', doc='', subblocks=(
        SubBlock('11.1', 'Métriques', settings=(
            S('ENABLE_METRICS', BOOL, default='true', scope=DEPLOY, bool_style='truefalse', apply=RESTART, doc='Métriques exposées sur /metrics si prometheus-fastapi-instrumentator est\ninstallé.'),
            S('PROMETHEUS_PUSHGATEWAY_URL', URL, default='', example='http://localhost:9091', render='commented', doc="Pushgateway des scripts CLI (vide = pas d'envoi) et nom du job."),
            S('METRICS_JOB_NAME', STR, default='ragpy_cli'),
            S('GRAFANA_PASSWORD', SECRET, default='', render='commented', scope=DEPLOY, apply=RECREATE, default_rule='admin (service Grafana commenté dans docker-compose.yml)', doc='Mot de passe admin de Grafana si le service est décommenté dans\ndocker-compose.yml (défaut : admin).'),
        )),
    )),
)
"""Blocs et sous-blocs, dans l'ordre du ``.env``."""


def _flatten(layout: Tuple[Block, ...]) -> Tuple[Setting, ...]:
    """Liste ordonnée des variables, avec ``block`` et ``subblock`` renseignés."""
    flat = []
    for block in layout:
        for sub in block.subblocks:
            for setting in sub.settings:
                flat.append(replace(setting, block=block.key, subblock=sub.key))
    return tuple(flat)


SETTINGS: Tuple[Setting, ...] = _flatten(LAYOUT)
"""Toutes les variables, dans l'ordre du ``.env``."""

BY_NAME: Mapping[str, Setting] = MappingProxyType({s.name: s for s in SETTINGS})
"""Accès par nom."""

INTERNAL_NAMES = frozenset({
    "ALBERT_LIVE",          # interrupteur des tests live : shell seulement, jamais dans un .env
    "RAGPY_DOTENV_DENY",    # écrit par build_subprocess_env, lu par rad_env
    "RAGPY_UPDATE_GOLDEN",  # régénération volontaire des goldens (tests)
    "TIKTOKEN_CACHE_DIR",   # cache tiktoken (conftest, CI)
    "DATA_GYM_CACHE_DIR",
    "UPLOADS_DIR",          # lu par le seul nettoyage : l'exposer le désaccorderait du reste
    "RAGPY_ENV_FILE",       # autre fichier .env (tests, conteneurs) : scripts/rad_settings/access.py
    "RAGPY_SETTINGS_STRICT",  # 0 = démarrage strict désactivé (tests, dépannage)
    "PATH",
    "HOSTNAME",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "PYTHONDONTWRITEBYTECODE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
})
"""Variables lues par le code ou l'outillage mais qui n'ont pas leur place dans
un ``.env`` (système, tests, interne)."""

_REMOVED_RAG = "fonction retirée le 2026-10-03 (réponses sourcées et rerank)"

OBSOLETE_NAMES: Mapping[str, str] = MappingProxyType({
    "MAX_ACTIVE_SESSIONS": "jamais lue par le code",
    "ALBERT_RERANK_RPM": _REMOVED_RAG,
    "ALBERT_RERANK_CONCURRENCY": _REMOVED_RAG,
    "ALBERT_TIMEOUT_RERANK": _REMOVED_RAG,
    "ALBERT_RAG_SEARCH_LIMIT": _REMOVED_RAG,
    "ALBERT_RAG_FINAL_LIMIT": _REMOVED_RAG,
    "ALBERT_RAG_CONTEXT_TOKENS": _REMOVED_RAG,
    "ALBERT_RAG_MAX_TOKENS": _REMOVED_RAG,
    "ALBERT_RAG_TEMPERATURE": _REMOVED_RAG,
    "ALBERT_RAG_TIMEOUT": _REMOVED_RAG,
    "ALBERT_RAG_MAX_ACTIVE": _REMOVED_RAG,
    "RERANK_SERVER": _REMOVED_RAG,
    "RERANK_MODEL": _REMOVED_RAG,
    "ALBERT_RERANK_ENABLED": _REMOVED_RAG,
    "ALBERT_RERANK_MODEL": _REMOVED_RAG,
    "LLM_RAG_SERVER": _REMOVED_RAG,
    "LLM_RAG_MODEL": _REMOVED_RAG,
    "ALBERT_RAG_ENABLED": _REMOVED_RAG,
    "ALBERT_RAG_MODEL": _REMOVED_RAG,
})
"""Variables rencontrées dans des ``.env`` existants que le code ne lit pas :
``env_tool tidy`` les range dans un bloc final « OBSOLÈTES »."""


def iter_layout() -> Iterator[Tuple[Block, SubBlock, Setting]]:
    """Parcourt le registre dans l'ordre : ``(bloc, sous-bloc, variable)``."""
    for block in LAYOUT:
        for sub in block.subblocks:
            for setting in sub.settings:
                yield block, sub, BY_NAME[setting.name]


def get(name: str) -> Optional[Setting]:
    """Variable ``name`` du registre, ou ``None`` si elle n'y figure pas."""
    return BY_NAME.get(name)


def names(status: Optional[str] = None) -> Tuple[str, ...]:
    """Noms des variables dans l'ordre du registre, filtrés par ``status``."""
    return tuple(s.name for s in SETTINGS if status is None or s.status == status)


def classify(name: str) -> str:
    """Classe un nom trouvé dans un ``.env`` : ``registered``, ``internal``,
    ``obsolete`` ou ``unknown``."""
    if name in BY_NAME:
        return "registered"
    if name in INTERNAL_NAMES:
        return "internal"
    if name in OBSOLETE_NAMES:
        return "obsolete"
    return "unknown"


def _check() -> None:
    """Invariants vérifiés à l'import : noms uniques, valeurs admises."""
    seen: Dict[str, str] = {}
    for s in SETTINGS:
        if s.name in seen:
            raise RuntimeError(f"variable {s.name} déclarée deux fois ({seen[s.name]}, {s.subblock})")
        seen[s.name] = s.subblock
        if s.kind not in KINDS or s.scope not in SCOPES or s.apply not in APPLIES:
            raise RuntimeError(f"métadonnée inconnue pour {s.name}")
        if s.status not in STATUSES or s.render not in RENDERS:
            raise RuntimeError(f"statut ou rendu inconnu pour {s.name}")
        if s.kind == BOOL and s.bool_style not in BOOL_STYLES:
            raise RuntimeError(f"style de booléen absent pour {s.name}")
        if s.kind == ENUM and s.example_value and s.example_value not in s.choices:
            raise RuntimeError(f"valeur d'exemple hors choix pour {s.name}")
        if s.family and s.family not in FAMILIES:
            raise RuntimeError(f"famille inconnue pour {s.name}")
        if s.name in INTERNAL_NAMES or s.name in OBSOLETE_NAMES:
            raise RuntimeError(f"{s.name} est à la fois enregistrée et interne ou obsolète")


_check()
