# Documentation technique UI - RAGpy Frontend

**Date de création** : 2025-11-22  
**Version** : 1.0  
**Audience** : Développeurs frontend et fullstack

## Vue d'ensemble architecture

### Stack technique
- **Backend** : FastAPI (Python) avec templates Jinja2
- **Frontend** : HTML5/CSS3/JavaScript vanilla
- **Communication** : REST API + Server-Sent Events (SSE)
- **Styling** : CSS custom properties + responsive design
- **Assets** : Fichiers statiques servis par FastAPI

### Structure fichiers
```
app/
├── main.py                 # Serveur FastAPI avec endpoints UI/API
├── templates/
│   └── index.html         # Template principal (SPA-style)
├── static/
│   ├── style.css          # Styles globaux et composants
│   └── favicon.ico        # Icône application
└── utils/                 # Utilitaires backend (Zotero, etc.)
```

---

## Architecture des templates

### Template principal (`app/templates/index.html`)

**Structure HTML sémantique** :
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <!-- Meta, title, CSS, inline styles -->
</head>
<body>
  <!-- Settings button + modal -->
  <!-- Step tracker (progress bar) -->
  <!-- Global controls -->
  <!-- Upload section (ZIP vs CSV) -->
  <!-- DataFrame section -->
  <!-- Decision tree (RAG vs Zotero) -->
  <!-- JavaScript state management -->
</body>
</html>
```

**Points clés** :
- **Single Page Application** : Tout dans un fichier HTML
- **Progressive disclosure** : Sections déverrouillées au fur et à mesure
- **State management** : JavaScript vanilla avec objet global `appState`

---

## Composants UI détaillés

### 1. Settings Modal - Configuration API

```html
<!-- Bouton settings (top-right) -->
<button id="settingsBtn" onclick="document.getElementById('settingsModal').style.display='block'">⚙</button>

<!-- Modal avec tous les credentials -->
<div id="settingsModal" class="modal">
  <div class="modal-content">
    <!-- 13 champs de configuration API -->
    <input type="password" id="openai_api_key" placeholder="sk-...">
    <input type="password" id="openrouter_api_key" placeholder="sk-or-v1-...">
    <!-- ... autres providers -->
  </div>
</div>
```

**Fonctionnalités** :
- **Masquage sécurisé** : Affiche `sk-abc123##########` pour les clés existantes
- **Validation temps réel** : Feedback visuel sur les champs requis
- **Persistance** : Sauvegarde dans `.env` via `POST /save_credentials`

### 2. Step Tracker - Barre de progression

```html
<div id="stepTracker" class="step-tracker">
  <div id="step-upload" class="step completed">
    <div class="step-icon">✓</div>
    <span>Upload</span>
  </div>
  <!-- 3 autres étapes : Extraction, Traitement, Destination -->
</div>
```

**États dynamiques** :
- `locked` : Étape non accessible (gris, opacity 0.5)
- `active` : Étape en cours (bleu, animation pulse)
- `completed` : Étape terminée (vert, checkmark)

**CSS transitions** :
```css
.step {
  transition: all 0.3s ease;
  transform: translateY(0);
}
.step.active {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);
}
```

### 3. Upload Section - Ingestion multi-sources

**Option A - ZIP Zotero** :
```html
<div class="upload-option" style="border-left: 4px solid #1976d2;">
  <h3>📦 Upload ZIP (Zotero + PDFs)</h3>
  <input type="file" accept=".zip" id="zipFileInput">
  <button onclick="uploadZip()">Upload & Extract</button>
</div>
```

**Option B - CSV Direct** :
```html
<div class="upload-option" style="border-left: 4px solid #2e7d32;">
  <h3>📊 Upload CSV (Direct)</h3>
  <input type="file" accept=".csv" id="csvFileInput">
  <button onclick="uploadCsv()">Upload CSV</button>
</div>
```

**Différenciation visuelle** : Bordures colorées (bleu Zotero, vert CSV)

### 4. DataFrame Section - Extraction

```html
<div id="dataframe-section" class="main-section locked">
  <h2>📄 Extract Text & Metadata</h2>
  <button id="processDataframe" onclick="processDataframe()" disabled>
    Extract Text
  </button>
  <div id="progressWrapper" style="display:none;">
    <div class="progress-bar-container">
      <div id="progressBar" class="progress-bar"></div>
    </div>
  </div>
</div>
```

**Progress bar animée** :
```css
.progress-bar {
  background: repeating-linear-gradient(90deg,
    transparent, transparent 8px,
    rgba(255,255,255,0.3) 8px, rgba(255,255,255,0.3) 10px
  );
  animation: slide 2s linear infinite;
}
```

### 5. Decision Tree - Arbre de traitement

**Branche RAG** (pipeline complet) :
```html
<div class="decision-branch" id="ragBranch">
  <h3 style="color: #1976d2;">🤖 RAG Pipeline</h3>
  
  <!-- 4 sous-étapes séquentielles -->
  <div class="sub-step" id="chunking-step">
    <h4>3.1 Initial Text Chunking</h4>
    <select id="chunkingModel">
      <option value="gpt-4o-mini">OpenAI (gpt-4o-mini)</option>
      <option value="openai/gemini-2.5-flash">OpenRouter (économique)</option>
    </select>
    <button onclick="processChunking()">Generate Chunks</button>
  </div>
  
  <!-- 3.2 Dense, 3.3 Sparse, 3.4 Vector DB -->
</div>
```

**Branche Zotero** (notes académiques) :
```html
<div class="decision-branch" id="zoteroBranch">
  <h3 style="color: #9c27b0;">📚 Zotero Notes</h3>
  
  <select id="zoteroModel">
    <option value="gpt-4o-mini">OpenAI (gpt-4o-mini)</option>
    <option value="openai/gemini-2.5-flash">OpenRouter (économique)</option>
  </select>
  
  <label>
    <input type="checkbox" id="extendedAnalysis">
    Extended Analysis (8000-12000 mots)
  </label>
  
  <button onclick="generateZoteroNotes()">Generate Notes</button>
</div>
```

---

## Gestion d'état JavaScript

### État global application

```javascript
const appState = {
  currentPath: '',              // Session path (UUID)
  
  steps: {
    upload: { completed: false, file: null },
    extraction: { completed: false, file: 'output.csv' },
    chunking: { completed: false, file: 'output_chunks.json' },
    dense: { completed: false, file: 'output_chunks_with_embeddings.json' },
    sparse: { completed: false, file: 'output_chunks_with_embeddings_sparse.json' }
  },
  
  selectedBranches: {
    rag: true,                  // Pipeline RAG actif par défaut
    zotero: false               // Notes Zotero optionnelles
  },
  
  isZoteroExport: false         // Détection type d'export
};
```

### Fonctions de gestion d'état

**Mise à jour tracker** :
```javascript
function updateStepTracker(stepName, status) {
  const stepElement = document.getElementById(`step-${stepName}`);
  
  // Supprime anciens états
  stepElement.classList.remove('locked', 'active', 'completed');
  
  // Applique nouveau statut
  stepElement.classList.add(status);
  
  if (status === 'completed') {
    stepElement.querySelector('.step-icon').textContent = '✓';
  }
}
```

**Déverrouillage sections** :
```javascript
function unlockSection(sectionId) {
  const section = document.getElementById(sectionId);
  section.classList.remove('locked');
  section.classList.add('step-available', 'fade-in');
  
  // Active les boutons dans la section
  const buttons = section.querySelectorAll('button');
  buttons.forEach(btn => btn.disabled = false);
}
```

**Contrôle branches** :
```javascript
function toggleBranch(branchName, enabled) {
  appState.selectedBranches[branchName] = enabled;
  
  const branchElement = document.getElementById(`${branchName}Branch`);
  branchElement.style.opacity = enabled ? '1' : '0.6';
  branchElement.style.borderColor = enabled ? 'initial' : '#ccc';
}
```

---

## API Communication patterns

### 1. Upload avec FormData

```javascript
async function uploadZip() {
  const fileInput = document.getElementById('zipFileInput');
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  
  try {
    const response = await fetch('/upload_zip', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    if (response.ok) {
      appState.currentPath = result.path;
      updateStepTracker('upload', 'completed');
      unlockSection('dataframe-section');
    }
  } catch (error) {
    console.error('Upload failed:', error);
  }
}
```

### 2. Server-Sent Events (SSE) pour progress

```javascript
async function processDataframe() {
  const formData = new FormData();
  formData.append('path', appState.currentPath);
  
  try {
    const response = await fetch('/process_dataframe_sse', {
      method: 'POST',
      body: formData
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));
          handleSSEMessage(data);
        }
      }
    }
  } catch (error) {
    console.error('Processing failed:', error);
  }
}

function handleSSEMessage(data) {
  switch (data.type) {
    case 'init':
      updateProgress(0, data.total, 'Initializing...');
      break;
    case 'progress':
      updateProgress(data.current, data.total, `Processing ${data.current}/${data.total}`);
      break;
    case 'complete':
      updateStepTracker('extraction', 'completed');
      unlockSection('decision-tree');
      break;
    case 'error':
      showError(data.message);
      break;
  }
}
```

### 3. Configuration credentials

```javascript
async function loadCredentials() {
  try {
    const response = await fetch('/get_credentials');
    const credentials = await response.json();
    
    // Populate form fields avec masquage sécurisé
    Object.entries(credentials).forEach(([key, value]) => {
      const input = document.getElementById(key.toLowerCase());
      if (input && value) {
        // Masque les clés : affiche 20 premiers chars + ##########
        const maskedValue = value.length > 20 
          ? value.substring(0, 20) + '##########'
          : value;
        input.value = maskedValue;
        input.dataset.originalValue = value; // Stockage valeur complète
      }
    });
  } catch (error) {
    console.error('Failed to load credentials:', error);
  }
}

async function saveCredentials() {
  const formData = {};
  
  // Collecte tous les champs credentials
  document.querySelectorAll('#settingsForm input').forEach(input => {
    const key = input.id.toUpperCase();
    let value = input.value;
    
    // Si masqué et non modifié, utilise valeur originale
    if (value.includes('##########') && input.dataset.originalValue) {
      value = input.dataset.originalValue;
    }
    
    formData[key] = value;
  });
  
  try {
    const response = await fetch('/save_credentials', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    if (response.ok) {
      showSuccess('Credentials saved successfully!');
    }
  } catch (error) {
    showError('Failed to save credentials');
  }
}
```

---

## Design System & CSS

### Variables CSS

```css
:root {
  /* Couleurs principales */
  --primary-blue: #1976d2;
  --primary-green: #2e7d32;
  --primary-purple: #9c27b0;
  
  /* États */
  --success-color: #4caf50;
  --warning-color: #ff9800;
  --error-color: #f44336;
  
  /* Espacements */
  --spacing-sm: 8px;
  --spacing-md: 16px;
  --spacing-lg: 24px;
  
  /* Transitions */
  --transition-fast: 0.2s ease;
  --transition-normal: 0.3s ease;
}
```

### Composants réutilisables

```css
/* Boutons primaires */
.btn-primary {
  background: var(--primary-blue);
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 8px;
  cursor: pointer;
  transition: var(--transition-normal);
}

.btn-primary:hover:not(:disabled) {
  background: #1565c0;
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(25, 118, 210, 0.3);
}

/* Sections principales */
.main-section {
  background: white;
  border-radius: 12px;
  padding: var(--spacing-lg);
  margin: var(--spacing-md) 0;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  transition: var(--transition-normal);
}

.main-section.locked {
  opacity: 0.5;
  pointer-events: none;
}

/* Animations d'entrée */
.fade-in {
  animation: fadeIn 0.5s ease-in-out;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### Responsive design

```css
/* Mobile-first approach */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 16px;
}

/* Tablet */
@media (min-width: 768px) {
  .decision-tree {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: var(--spacing-lg);
  }
}

/* Desktop */
@media (min-width: 1024px) {
  .upload-options {
    display: flex;
    gap: var(--spacing-lg);
  }
  
  .upload-option {
    flex: 1;
  }
}
```

---

## Patterns avancés

### 1. Upload par drag & drop (futur)

```javascript
function setupDragDrop(elementId, callback) {
  const element = document.getElementById(elementId);
  
  element.addEventListener('dragover', (e) => {
    e.preventDefault();
    element.classList.add('drag-over');
  });
  
  element.addEventListener('dragleave', () => {
    element.classList.remove('drag-over');
  });
  
  element.addEventListener('drop', (e) => {
    e.preventDefault();
    element.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    callback(files);
  });
}
```

### 2. Toast notifications système

```javascript
function showToast(message, type = 'info', duration = 3000) {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  
  document.body.appendChild(toast);
  
  // Animation d'entrée
  setTimeout(() => toast.classList.add('show'), 100);
  
  // Auto-dismiss
  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => document.body.removeChild(toast), 300);
  }, duration);
}
```

### 3. Validation formulaires temps réel

```javascript
function setupFormValidation(formId) {
  const form = document.getElementById(formId);
  
  form.querySelectorAll('input[required]').forEach(input => {
    input.addEventListener('input', () => {
      validateField(input);
    });
    
    input.addEventListener('blur', () => {
      validateField(input);
    });
  });
}

function validateField(input) {
  const isValid = input.checkValidity();
  
  input.classList.toggle('invalid', !isValid);
  input.classList.toggle('valid', isValid);
  
  // Feedback visuel avec message
  const feedback = input.nextElementSibling;
  if (feedback && feedback.classList.contains('feedback')) {
    feedback.textContent = isValid ? '' : input.validationMessage;
  }
}
```

---

## Recommandations d'évolution

### 1. Architecture moderne
- **Vue.js/React** : Composants réutilisables et state management
- **TypeScript** : Type safety et meilleure DX
- **Vite/Webpack** : Bundling et optimisation

### 2. UX/UI améliorés
- **Design system** : Token-based avec Figma
- **Animations** : Framer Motion ou CSS advanced
- **Accessibilité** : ARIA labels, keyboard navigation
- **PWA** : Service workers, offline support

### 3. Performance
- **Lazy loading** : Images et sections non critiques
- **Code splitting** : Charge uniquement le nécessaire
- **CDN** : Assets statiques optimisés
- **Caching** : Stratégies intelligentes

### 4. Monitoring
- **Error tracking** : Sentry pour crash reporting
- **Analytics** : User journey et conversion funnel
- **Performance** : Core Web Vitals monitoring
- **A/B testing** : Optimisation continue UI/UX

Cette documentation fournit une base solide pour comprendre, maintenir et faire évoluer l'interface utilisateur de RAGpy vers une architecture frontend moderne.