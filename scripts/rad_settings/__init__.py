"""Configuration unifiée de RAGpy : registre des variables et outils du ``.env``.

Sprint « configuration unifiée » (``.claude/tasks/SPRINT_config_unifiee.md``).
Paquet de la bibliothèque standard seulement, importable en
``scripts.rad_settings`` (application, tests) et en ``rad_settings`` (scripts
CLI lancés avec ``scripts/`` sur ``sys.path``). Il ne réexporte rien : importer
explicitement ``registry``, ``models`` ou ``dotenv_file``.
"""
