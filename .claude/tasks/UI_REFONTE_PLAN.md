# Plan de Développement : Refonte de l'Interface UI RAGpy

**Date**: 2025-11-22
**Statut**: En cours

---

## Objectifs

1. **Étapes toujours visibles** : Toutes les étapes affichées dès le début (plus de révélation progressive)
2. **Indicateurs visuels** : Bleu (en attente) → Vert (complété avec lien vers fichier produit)
3. **Arbre décisionnel** : Après CSV, bifurcation entre RAG et Zotero Notes

---

## Architecture Actuelle

- **Frontend**: `app/templates/index.html` (979 lignes)
- **Styles**: `app/static/style.css` (156 lignes)
- **Backend**: `app/main.py` (1293 lignes) - Pas de modification nécessaire

### Mécanisme actuel de révélation
Les sections utilisent `style="display:none"` et sont révélées via JavaScript après chaque étape.

---

## Phase 1 : Nouveau système d'étapes visuelles

### 1.1 Step Tracker Component

Barre horizontale en haut de page montrant toutes les étapes :

```
[1. Upload] → [2. Extraction] → [3. Traitement] → [4. Destination]
   🔵            ⬜               ⬜                 ⬜
```

Légende :
- 🔵 Bleu : Étape active/en attente
- ✅ Vert : Étape complétée (avec lien téléchargement)
- ⬜ Gris : Étape verrouillée

### 1.2 Structure HTML du Step Tracker

```html
<div class="step-tracker">
  <div class="step-item active" data-step="1">
    <span class="step-number">1</span>
    <span class="step-title">Upload</span>
    <a class="step-file-link" style="display:none"></a>
  </div>
  <div class="step-connector"></div>
  <div class="step-item locked" data-step="2">
    <span class="step-number">2</span>
    <span class="step-title">Extraction</span>
    <a class="step-file-link" style="display:none"></a>
  </div>
  <!-- ... autres étapes ... -->
</div>
```

### 1.3 CSS pour les états

```css
/* État par défaut - verrouillé */
.step-item {
  opacity: 0.5;
  color: #9e9e9e;
}

/* État actif - bleu */
.step-item.active {
  opacity: 1;
  color: #1976d2;
  border-color: #1976d2;
}

/* État complété - vert */
.step-item.completed {
  opacity: 1;
  color: #4caf50;
  border-color: #4caf50;
}

/* Lien fichier */
.step-file-link {
  font-size: 0.8em;
  color: #4caf50;
}
```

---

## Phase 2 : Arbre de bifurcation POST-CSV

### 2.1 Diagramme de flux

```
┌─────────────────────────────────────────┐
│  Étape 1: Upload (ZIP ou CSV)           │
│  Étape 2: Extraction texte (si ZIP)     │
└─────────────────┬───────────────────────┘
                  │
        ┌─────────▼─────────┐
        │   CSV PRÊT        │
        │ (output.csv)      │
        └────────┬──────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───▼───────────────┐   ┌─────▼─────────────┐
│   🔍 BRANCHE RAG  │   │   📝 BRANCHE      │
│     (gauche)      │   │   ZOTERO NOTES    │
│                   │   │     (droite)      │
│ • 3.1 Chunking    │   │                   │
│ • 3.2 Dense Embed │   │ • Génération LLM  │
│ • 3.3 Sparse Embed│   │ • Push vers Zotero│
│ • 4. Vector DB    │   │                   │
└───────────────────┘   └───────────────────┘
```

### 2.2 Structure HTML de l'arbre

```html
<!-- Apparaît après CSV prêt -->
<div class="decision-tree" id="decision-tree" style="display:none">
  <h2>Choisissez votre destination</h2>
  <p class="tree-subtitle">Vous pouvez sélectionner l'une ou les deux options</p>

  <div class="branches-container">
    <!-- Branche gauche : RAG -->
    <div class="branch branch-rag" id="rag-branch">
      <div class="branch-header">
        <input type="checkbox" id="rag-checkbox" checked>
        <label for="rag-checkbox">
          <span class="branch-icon">🔍</span>
          <span class="branch-title">Nourrir un RAG</span>
        </label>
      </div>
      <div class="branch-content">
        <!-- Étapes 3.1, 3.2, 3.3, 4 -->
        <section id="initial-chunk-section">...</section>
        <section id="dense-embedding-section">...</section>
        <section id="sparse-embedding-section">...</section>
        <section id="vector-db-section">...</section>
      </div>
    </div>

    <!-- Branche droite : Zotero -->
    <div class="branch branch-zotero" id="zotero-branch">
      <div class="branch-header">
        <input type="checkbox" id="zotero-checkbox">
        <label for="zotero-checkbox">
          <span class="branch-icon">📝</span>
          <span class="branch-title">Notes Zotero</span>
        </label>
      </div>
      <div class="branch-content">
        <!-- Génération notes Zotero -->
        <section id="zotero-notes-section">...</section>
      </div>
    </div>
  </div>
</div>
```

### 2.3 CSS pour l'arbre

```css
.decision-tree {
  margin: 30px 0;
  padding: 20px;
  background: #f5f5f5;
  border-radius: 12px;
}

.branches-container {
  display: flex;
  gap: 20px;
  margin-top: 20px;
}

.branch {
  flex: 1;
  background: white;
  border-radius: 8px;
  padding: 20px;
  border: 2px solid #e0e0e0;
  transition: border-color 0.3s;
}

.branch.selected {
  border-color: #1976d2;
  box-shadow: 0 2px 8px rgba(25, 118, 210, 0.2);
}

.branch-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding-bottom: 15px;
  border-bottom: 1px solid #e0e0e0;
}

.branch-icon {
  font-size: 1.5em;
}

.branch-title {
  font-size: 1.2em;
  font-weight: 600;
}

.branch-content {
  padding-top: 15px;
}

/* Branche désactivée */
.branch:not(.selected) .branch-content {
  opacity: 0.5;
  pointer-events: none;
}
```

---

## Phase 3 : JavaScript - Gestion d'état

### 3.1 Nouvel objet state

```javascript
const appState = {
  currentPath: '',
  steps: {
    upload: { completed: false, file: null },
    extraction: { completed: false, file: 'output.csv' },
    chunking: { completed: false, file: 'output_chunks.json' },
    denseEmbed: { completed: false, file: 'output_chunks_with_embeddings.json' },
    sparseEmbed: { completed: false, file: 'output_chunks_with_embeddings_sparse.json' },
    vectorDb: { completed: false, file: null },
    zoteroNotes: { completed: false, file: null }
  },
  selectedBranches: {
    rag: true,
    zotero: false
  }
};
```

### 3.2 Fonctions de mise à jour

```javascript
function completeStep(stepName, fileName) {
  appState.steps[stepName].completed = true;
  appState.steps[stepName].file = fileName;
  updateStepTracker();
  updateSectionStyles();
}

function updateStepTracker() {
  Object.entries(appState.steps).forEach(([name, data]) => {
    const stepEl = document.querySelector(`[data-step="${name}"]`);
    if (!stepEl) return;

    stepEl.classList.remove('active', 'completed', 'locked');

    if (data.completed) {
      stepEl.classList.add('completed');
      if (data.file) {
        const link = stepEl.querySelector('.step-file-link');
        link.href = buildDownloadLink(data.file);
        link.textContent = `📎 ${data.file}`;
        link.style.display = 'inline';
      }
    } else if (isStepAvailable(name)) {
      stepEl.classList.add('active');
    } else {
      stepEl.classList.add('locked');
    }
  });
}

function showDecisionTree() {
  document.getElementById('decision-tree').style.display = 'block';
  // Animation d'apparition
  document.getElementById('decision-tree').classList.add('fade-in');
}

function toggleBranch(branchName, enabled) {
  appState.selectedBranches[branchName] = enabled;
  const branch = document.getElementById(`${branchName}-branch`);
  branch.classList.toggle('selected', enabled);
}
```

---

## Phase 4 : Fichiers à modifier

| Fichier | Modifications | Lignes estimées |
|---------|---------------|-----------------|
| `app/static/style.css` | Ajout styles step-tracker, decision-tree, couleurs | +80 lignes |
| `app/templates/index.html` | Restructuration complète, nouveau layout | ~200 lignes modifiées |
| `app/main.py` | Aucune modification nécessaire | 0 |

---

## Phase 5 : Tests

### Parcours à tester

- [ ] ZIP → Extraction → RAG complet (chunking → embeddings → vector DB)
- [ ] ZIP → Extraction → Zotero Notes
- [ ] ZIP → Extraction → RAG + Zotero (les deux branches)
- [ ] CSV direct → RAG complet
- [ ] CSV direct → Zotero Notes
- [ ] CSV direct → RAG + Zotero

### Points de vérification

- [ ] Step tracker affiche correctement les états bleu/vert
- [ ] Liens de téléchargement fonctionnels après complétion
- [ ] Arbre de décision apparaît après CSV prêt
- [ ] Branches peuvent être activées/désactivées indépendamment
- [ ] Responsive design sur mobile/tablette

---

## Ordre d'implémentation

1. ✅ Créer ce fichier de plan
2. ✅ Modifier `style.css` : ajouter styles step-tracker et decision-tree
3. ✅ Modifier `index.html` : ajouter Step Tracker en haut
4. ✅ Modifier `index.html` : restructurer sections pour tout afficher
5. ✅ Modifier `index.html` : créer l'arbre de décision bifurcation
6. ✅ Modifier `index.html` : adapter JavaScript pour nouveau state
7. ⬜ Tests complets

---

## Notes techniques

### Compatibilité
- Pas de framework JS nécessaire (vanilla JS)
- CSS flexbox pour le layout de l'arbre
- Compatible avec le backend FastAPI existant

### Points d'attention
- Conserver la logique de session (`currentPath`)
- Maintenir les fallback uploads pour chaque étape
- Préserver la gestion des credentials dans la modal settings
