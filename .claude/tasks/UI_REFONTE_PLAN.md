# Plan de refonte UI - RAGpy Interface Professionnelle

**Date de création** : 2025-11-22  
**Objectif** : Migration vers une interface moderne tout en préservant l'ergonomie actuelle  
**Scope** : Refonte complète frontend avec architecture scalable pour futures évolutions  

## Vision produit

### Objectifs principaux
1. **Modernisation technologique** : Migration vers stack moderne (Vue.js 3 + TypeScript)
2. **Design professionnel** : Interface niveau entreprise avec design system cohérent  
3. **Préservation ergonomie** : Maintenir la simplicité d'usage actuelle (UX inchangée)
4. **Extensibilité** : Architecture prête pour gestion utilisateurs et nouvelles fonctionnalités
5. **Performance** : Optimisation loading, responsivité et expérience utilisateur

### Contraintes critiques
✅ **MAINTENIR** l'ergonomie step-by-step actuelle  
✅ **PRÉSERVER** le flux utilisateur ZIP → Extraction → Pipeline  
✅ **GARDER** la simplicité : "Upload → Process → Result"  
✅ **COMPATIBILITÉ** avec les API endpoints FastAPI existants  

---

## Analyse de l'existant - Points forts à préserver

### 🎯 **Ergonomie excellente à maintenir**
- **Progressive disclosure** : Sections déverrouillées progressivement
- **Step tracker visuel** : 4 étapes claires (Upload → Extract → Process → Destination)
- **Decision tree intuitive** : Choix RAG vs Zotero clairement séparés
- **Feedback temps réel** : SSE progress bars pour opérations longues
- **Upload simplifié** : ZIP/CSV avec différenciation visuelle claire

### 🔧 **Fonctionnalités core à migrer**
- Settings modal avec 13 providers API
- Upload dual : ZIP (Zotero+PDFs) / CSV (direct)
- Pipeline branché : RAG complet vs Notes Zotero
- Progress tracking SSE en temps réel
- Download des artifacts générés

### ⚠️ **Limitations actuelles à corriger**
- Code JavaScript monolithique (1500+ lignes dans HTML)
- Pas de composants réutilisables
- State management artisanal
- CSS inline et styles éparpillés
- Pas de validation robuste côté client
- Interface non responsive sur mobile
- Accessibilité limitée

---

## Architecture cible

### Stack technologique moderne

```
Frontend (Nouveau)
├── Vue.js 3 + Composition API    # Framework réactif moderne
├── TypeScript                    # Type safety + meilleure DX
├── Vite                         # Build tool ultra-rapide
├── Pinia                        # State management Vue 3
├── Vue Router                   # SPA routing pour futurs modules
├── Vuelidate                    # Validation formulaires
├── Headless UI                  # Composants accessibles
└── Tailwind CSS                 # Design system utilitaire

Backend (Inchangé mais étendu)
├── FastAPI                      # API REST + SSE (existant)
├── Jinja2 templates            # Supprimé (remplacé par SPA)
└── Static files               # Servira uniquement les assets build
```

### Architecture composants

```
src/
├── components/                  # Composants réutilisables
│   ├── ui/                     # Design system base
│   │   ├── Button.vue          # Boutons avec variants
│   │   ├── Card.vue            # Cartes et containers
│   │   ├── Modal.vue           # Modales accessibles
│   │   ├── ProgressBar.vue     # Barres de progression
│   │   └── Notification.vue    # Toast system
│   ├── layout/                 # Structure application
│   │   ├── AppHeader.vue       # Header avec navigation
│   │   ├── AppSidebar.vue      # Sidebar (futur multi-projets)
│   │   └── StepTracker.vue     # Tracker progression
│   └── features/               # Composants métier
│       ├── Upload/             # Zone upload
│       ├── Pipeline/           # Traitement RAG
│       ├── Settings/           # Configuration
│       └── Zotero/             # Intégration Zotero
├── composables/                # Logic réutilisable
│   ├── useApi.ts              # Client API avec types
│   ├── useSSE.ts              # Server-Sent Events
│   ├── useUpload.ts           # Gestion uploads
│   └── useSteps.ts            # State machine steps
├── stores/                     # State management Pinia
│   ├── app.ts                 # État global application
│   ├── pipeline.ts            # État pipeline RAG
│   ├── settings.ts            # Configuration utilisateur
│   └── auth.ts                # Authentification (futur)
├── types/                      # Types TypeScript
├── utils/                      # Utilitaires
└── views/                      # Pages/Routes
    ├── Dashboard.vue          # Interface principale
    ├── Login.vue              # Authentification (futur)
    └── Projects.vue           # Multi-projets (futur)
```

---

## Phase 1 : Migration fondations (3-4 semaines)

### Objectif : Reproduire l'interface actuelle en Vue.js

#### Semaine 1 : Setup projet et design system

**Setup technique** :
```bash
# Initialisation projet Vue.js
npm create vue@latest ragpy-frontend
cd ragpy-frontend
npm install

# Dependencies core
npm install @vue/typescript @vueuse/core pinia vue-router

# UI/UX
npm install @headlessui/vue @tailwindcss/forms
npm install lucide-vue-next  # Icons modernes

# Build et dev tools
npm install vite @vitejs/plugin-vue typescript
```

**Design system base** :
- **Variables design** : Migration variables CSS vers Tailwind config
- **Composants UI primitifs** : Button, Card, Input, Modal, ProgressBar
- **Tokens couleurs** : Système cohérent (primary-blue, success-green, etc.)
- **Typography scale** : Hiérarchie textes et espacements
- **Dark mode ready** : CSS custom properties préparées

**Delivrables** :
- [ ] Setup Vite + Vue 3 + TypeScript fonctionnel
- [ ] Storybook pour composants UI (développement isolé)
- [ ] Tailwind config avec tokens design RAGpy
- [ ] 8-10 composants UI de base documentés

#### Semaine 2 : Architecture state et API

**State management** :
```typescript
// stores/app.ts
export const useAppStore = defineStore('app', {
  state: () => ({
    currentPath: '',
    isLoading: false,
    notifications: [] as Notification[]
  }),
  
  actions: {
    setCurrentPath(path: string) {
      this.currentPath = path
    },
    showNotification(message: string, type: 'success' | 'error') {
      // Implementation toast système
    }
  }
})

// stores/pipeline.ts  
export const usePipelineStore = defineStore('pipeline', {
  state: () => ({
    steps: {
      upload: { completed: false, file: null },
      extraction: { completed: false, file: 'output.csv' },
      chunking: { completed: false, file: 'output_chunks.json' },
      // ...
    } as PipelineSteps,
    
    selectedBranches: {
      rag: true,
      zotero: false
    }
  }),
  
  getters: {
    currentStep(): StepName {
      // Logic déterminant l'étape active
    },
    canProceedToNextStep(): boolean {
      // Validation progression
    }
  }
})
```

**API Client typé** :
```typescript
// composables/useApi.ts
import type { UploadResponse, PipelineStep, Settings } from '@/types'

export function useApi() {
  const uploadZip = async (file: File): Promise<UploadResponse> => {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await fetch('/upload_zip', {
      method: 'POST',
      body: formData
    })
    
    if (!response.ok) throw new Error('Upload failed')
    return response.json()
  }
  
  const processDataframeSSE = (path: string, onProgress: (data: any) => void) => {
    // Implementation SSE avec types
  }
  
  return { uploadZip, processDataframeSSE }
}
```

**Delivrables** :
- [ ] Stores Pinia complets avec types TypeScript
- [ ] API client avec tous les endpoints typés
- [ ] SSE composable pour progress temps réel
- [ ] Error handling et notifications centralisées

#### Semaine 3 : Composants métier principaux

**Upload Component** :
```vue
<!-- components/features/Upload/UploadZone.vue -->
<template>
  <div class="grid md:grid-cols-2 gap-6">
    <!-- Option ZIP -->
    <UploadCard
      icon="📦"
      title="Upload ZIP (Zotero + PDFs)"
      description="Export Zotero avec PDFs attachés"
      color="blue"
      accept=".zip"
      @upload="handleZipUpload"
    />
    
    <!-- Option CSV -->
    <UploadCard
      icon="📊"
      title="Upload CSV (Direct)"
      description="Données structurées sans OCR"
      color="green"
      accept=".csv"
      @upload="handleCsvUpload"
    />
  </div>
</template>

<script setup lang="ts">
import { useUpload } from '@/composables/useUpload'
import { usePipelineStore } from '@/stores/pipeline'

const { uploadZip, uploadCsv } = useUpload()
const pipeline = usePipelineStore()

async function handleZipUpload(file: File) {
  try {
    const result = await uploadZip(file)
    pipeline.completeStep('upload', result.path)
  } catch (error) {
    // Error handling
  }
}
</script>
```

**Step Tracker Component** :
```vue
<!-- components/layout/StepTracker.vue -->
<template>
  <div class="step-tracker">
    <StepItem
      v-for="(step, index) in steps"
      :key="step.name"
      :step="step"
      :index="index"
      :is-active="currentStep === step.name"
      :is-completed="step.completed"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { usePipelineStore } from '@/stores/pipeline'

const pipeline = usePipelineStore()
const steps = computed(() => pipeline.stepsArray)
const currentStep = computed(() => pipeline.currentStep)
</script>
```

**Delivrables** :
- [ ] UploadZone avec drag&drop et validation
- [ ] StepTracker avec animations et états visuels
- [ ] ProcessingSection avec progress SSE intégré
- [ ] SettingsModal avec validation formulaire Vuelidate

#### Semaine 4 : Pipeline et intégration

**Decision Tree Component** :
```vue
<!-- components/features/Pipeline/DecisionTree.vue -->
<template>
  <div class="decision-tree grid lg:grid-cols-2 gap-8">
    <!-- Branche RAG -->
    <PipelineBranch
      name="rag"
      title="🤖 RAG Pipeline"
      color="blue"
      :steps="ragSteps"
      :active="selectedBranches.rag"
      @toggle="toggleBranch"
    />
    
    <!-- Branche Zotero -->
    <PipelineBranch
      name="zotero"
      title="📚 Zotero Notes"
      color="purple"
      :steps="zoteroSteps"
      :active="selectedBranches.zotero"
      @toggle="toggleBranch"
    />
  </div>
</template>
```

**SSE Integration** :
```typescript
// composables/useSSE.ts
export function useSSE() {
  const connect = (endpoint: string, onMessage: (data: any) => void) => {
    return new Promise((resolve, reject) => {
      fetch(endpoint, { method: 'POST' })
        .then(response => {
          const reader = response.body?.getReader()
          const decoder = new TextDecoder()
          
          const readChunk = () => {
            reader?.read().then(({ done, value }) => {
              if (done) {
                resolve(undefined)
                return
              }
              
              const chunk = decoder.decode(value)
              const lines = chunk.split('\n')
              
              lines.forEach(line => {
                if (line.startsWith('data: ')) {
                  const data = JSON.parse(line.slice(6))
                  onMessage(data)
                }
              })
              
              readChunk()
            })
          }
          
          readChunk()
        })
        .catch(reject)
    })
  }
  
  return { connect }
}
```

**Delivrables** :
- [ ] DecisionTree avec branches configurables
- [ ] Pipeline steps avec sub-components
- [ ] SSE integration complète et robuste
- [ ] Interface 100% fonctionnelle (parité avec l'actuelle)

---

## Phase 2 : Amélioration design et UX (2-3 semaines)

### Objectif : Interface niveau professionnel avec design moderne

#### Semaine 5-6 : Design system avancé

**Design tokens étendus** :
```typescript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#e3f2fd',
          100: '#bbdefb',
          500: '#1976d2',
          600: '#1565c0',
          700: '#0d47a1'
        },
        success: {
          50: '#e8f5e8',
          500: '#4caf50',
          600: '#43a047'
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif']
      },
      boxShadow: {
        'card': '0 2px 8px rgba(0,0,0,0.1)',
        'card-hover': '0 8px 32px rgba(0,0,0,0.12)'
      }
    }
  }
}
```

**Composants UI avancés** :
```vue
<!-- components/ui/Card.vue -->
<template>
  <div
    :class="[
      'rounded-xl transition-all duration-300',
      'bg-white dark:bg-gray-800',
      'shadow-card hover:shadow-card-hover',
      {
        'border-l-4 border-primary-500': variant === 'primary',
        'border-l-4 border-success-500': variant === 'success',
        'opacity-60 pointer-events-none': disabled
      }
    ]"
  >
    <div class="p-6">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'success' | 'default'
  disabled?: boolean
}

withDefaults(defineProps<Props>(), {
  variant: 'default',
  disabled: false
})
</script>
```

**Animations et transitions** :
```css
/* Micro-interactions */
@keyframes slideInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes pulse-success {
  0% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0.4); }
  70% { box-shadow: 0 0 0 10px rgba(76, 175, 80, 0); }
  100% { box-shadow: 0 0 0 0 rgba(76, 175, 80, 0); }
}

.step-completed {
  animation: pulse-success 0.6s ease-out;
}
```

**Delivrables** :
- [ ] Design system complet avec 30+ composants
- [ ] Dark mode fonctionnel avec toggle
- [ ] Animations fluides et micro-interactions
- [ ] Icons library (Lucide) intégrée
- [ ] Typography et spacing cohérents

#### Semaine 7 : UX et responsivité

**Responsive design avancé** :
```vue
<template>
  <!-- Mobile: Stack vertical -->
  <div class="space-y-4 lg:space-y-0 lg:grid lg:grid-cols-2 lg:gap-8">
    
    <!-- Upload zone adaptive -->
    <div class="space-y-4 lg:space-y-6">
      <UploadCard v-for="option in uploadOptions" />
    </div>
    
    <!-- Sidebar mobile: bottom sheet -->
    <div class="lg:sticky lg:top-6">
      <ProgressPanel />
    </div>
  </div>
</template>
```

**Accessibilité WCAG 2.1** :
- Focus management avec Vue directives
- Screen reader support complet
- Keyboard navigation optimisée
- Contraste couleurs validé
- Aria labels sur tous les éléments interactifs

**Mobile optimization** :
- Touch-friendly buttons (min 44px)
- Swipe gestures pour navigation
- Modal mobile-first avec bottom sheets
- Upload files optimisé mobile

**Delivrables** :
- [ ] Interface parfaitement responsive (mobile → desktop)
- [ ] Accessibilité WCAG 2.1 Level AA
- [ ] Touch interactions optimisées
- [ ] Performance Lighthouse 90+ score

---

## Phase 3 : Fonctionnalités avancées (2-3 semaines)

### Objectif : Features niveau entreprise pour évolutivité

#### Semaine 8-9 : Architecture multi-utilisateurs

**Authentification foundation** :
```typescript
// stores/auth.ts
export const useAuthStore = defineStore('auth', {
  state: () => ({
    user: null as User | null,
    isAuthenticated: false,
    currentWorkspace: null as Workspace | null
  }),
  
  actions: {
    async login(credentials: LoginCredentials) {
      // Login logic
    },
    async logout() {
      // Cleanup session
    },
    async switchWorkspace(workspaceId: string) {
      // Multi-tenant support
    }
  }
})

// Router guards
router.beforeEach(async (to, from) => {
  const auth = useAuthStore()
  
  if (to.meta.requiresAuth && !auth.isAuthenticated) {
    return '/login'
  }
})
```

**Multi-projets architecture** :
```vue
<!-- views/Dashboard.vue -->
<template>
  <div class="dashboard-layout">
    <!-- Sidebar projects -->
    <AppSidebar>
      <ProjectsList />
      <UserMenu />
    </AppSidebar>
    
    <!-- Main content -->
    <main class="flex-1">
      <ProjectHeader />
      <RAGPipeline />
    </main>
  </div>
</template>
```

**Gestion sessions avancée** :
- Sessions persistantes par projet
- Autosave état pipeline
- Recovery après déconnexion
- Partage sessions entre utilisateurs

#### Semaine 10 : Notifications et monitoring

**Toast système avancé** :
```typescript
// composables/useNotifications.ts
export function useNotifications() {
  const notifications = ref<Notification[]>([])
  
  const notify = (
    message: string, 
    options: NotificationOptions = {}
  ) => {
    const notification: Notification = {
      id: crypto.randomUUID(),
      message,
      type: options.type || 'info',
      duration: options.duration || 4000,
      actions: options.actions || []
    }
    
    notifications.value.push(notification)
    
    if (notification.duration > 0) {
      setTimeout(() => {
        dismiss(notification.id)
      }, notification.duration)
    }
    
    return notification.id
  }
  
  return { notifications: readonly(notifications), notify, dismiss }
}
```

**Analytics et monitoring** :
```typescript
// composables/useAnalytics.ts
export function useAnalytics() {
  const trackEvent = (eventName: string, properties: Record<string, any>) => {
    // Tracking pour optimiser UX
    // Performance metrics
    // Error tracking
  }
  
  const trackPipelineStep = (step: string, duration: number) => {
    trackEvent('pipeline_step_completed', { step, duration })
  }
  
  return { trackEvent, trackPipelineStep }
}
```

**Delivrables** :
- [ ] Architecture auth prête (sans backend auth encore)
- [ ] Multi-projets UI foundation
- [ ] Système notifications avancé
- [ ] Analytics events intégrés

---

## Phase 4 : Performance et production (1-2 semaines)

### Objectif : Optimisation et déploiement production

#### Semaine 11 : Optimisation performance

**Code splitting** :
```typescript
// router/index.ts
const routes = [
  {
    path: '/',
    component: () => import('@/views/Dashboard.vue')
  },
  {
    path: '/settings',
    component: () => import('@/views/Settings.vue')
  },
  {
    path: '/projects/:id',
    component: () => import('@/views/Project.vue')
  }
]
```

**Lazy loading composants** :
```vue
<script setup lang="ts">
// Lazy load heavy components
const ZoteroNotes = defineAsyncComponent(() => 
  import('@/components/features/Zotero/NotesGenerator.vue')
)

const VectorDatabase = defineAsyncComponent(() => 
  import('@/components/features/Pipeline/VectorDatabase.vue')
)
</script>
```

**PWA configuration** :
```typescript
// vite.config.ts
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}']
      },
      manifest: {
        name: 'RAGpy - Academic Research Pipeline',
        short_name: 'RAGpy',
        description: 'Process academic documents with RAG pipeline',
        theme_color: '#1976d2'
      }
    })
  ]
})
```

**Optimisations Vite** :
- Bundle splitting intelligent
- Tree-shaking optimisé
- Image optimization (webp, lazy loading)
- CSS purging et minification

#### Semaine 12 : Déploiement et CI/CD

**Docker frontend** :
```dockerfile
# Dockerfile.frontend
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

**CI/CD Pipeline** :
```yaml
# .github/workflows/frontend.yml
name: Frontend CI/CD
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm ci
      - run: npm run lint
      - run: npm run test:unit
      - run: npm run build
```

**Delivrables** :
- [ ] Bundle optimisé < 500KB initial
- [ ] Lighthouse score 95+ (Performance, Accessibility, SEO)
- [ ] PWA fonctionnelle avec offline support
- [ ] CI/CD automatisé avec tests et déploiement

---

## Intégration Backend

### Modifications FastAPI minimales

**Nouveau endpoint pour SPA** :
```python
# app/main.py
from fastapi.staticfiles import StaticFiles

# Serve Vue.js build
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/")
async def serve_spa():
    return FileResponse("frontend/dist/index.html")

@app.get("/{full_path:path}")  
async def serve_spa_routes(full_path: str):
    # Serve index.html for all non-API routes (SPA routing)
    if full_path.startswith("api/"):
        raise HTTPException(404)
    return FileResponse("frontend/dist/index.html")
```

**API versioning** :
```python
# Gradual migration
@app.include_router(api_router, prefix="/api/v1")

# Legacy endpoints keep working
@app.post("/upload_zip")  # Existing
@app.post("/api/v1/upload")  # New typed endpoint
```

**CORS configuration** :
```python
# Production-ready CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ragpy.yourdomain.com"],  # Specific domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Migration strategy

### Déploiement progressif

**Phase 1 : Coexistence** (2 semaines)
- Nouvelle interface accessible sur `/v2` 
- Interface actuelle reste sur `/`
- Tests utilisateurs et feedback

**Phase 2 : A/B Testing** (1 semaine)
- 50% utilisateurs → nouvelle interface
- Monitoring performance et erreurs
- Ajustements basés sur retours

**Phase 3 : Migration complète** (1 semaine)
- Switch définitif vers nouvelle interface
- Interface legacy en backup
- Monitoring complet post-migration

### Rollback plan
- Interface legacy maintenue 1 mois
- Switch immediate possible via feature flag
- Monitoring erreurs en temps réel
- Hotfix capabilities maintenues

---

## Budget et timeline

### Timeline global : 12 semaines

| Phase | Durée | Effort | Priorité |
|-------|-------|---------|----------|
| **Migration fondations** | 4 semaines | 160h | CRITIQUE |
| **Design professionnel** | 3 semaines | 120h | HAUTE |
| **Features avancées** | 3 semaines | 120h | MOYENNE |
| **Production** | 2 semaines | 80h | CRITIQUE |

### Ressources nécessaires
- **1 Développeur Frontend Senior** (Vue.js/TypeScript expert)
- **1 UI/UX Designer** (pour design system et prototypes)
- **0.5 DevOps** (pour CI/CD et déploiement)

### Coût technologique
- Design tools (Figma Pro) : 15€/mois
- Hosting frontend (Vercel/Netlify) : 20€/mois  
- Monitoring (Sentry) : 26€/mois
- **Total récurrent** : ~60€/mois

---

## Critères de succès

### KPIs techniques
- **Performance** : Lighthouse 95+ (vs 70 actuel)
- **Accessibilité** : WCAG 2.1 AA compliance
- **Mobile** : Expérience native iOS/Android
- **Bundle size** : < 500KB initial load
- **Error rate** : < 0.1% client errors

### KPIs utilisateur  
- **Time to first interaction** : < 2s (vs 5s actuel)
- **Task completion rate** : 98%+ (maintenir niveau actuel)
- **User satisfaction** : 4.5/5 (vs 4.0 actuel)
- **Mobile adoption** : 40%+ sessions mobiles

### KPIs business
- **Development velocity** : +50% nouvelles features
- **Maintenance cost** : -30% bugs et hotfixes
- **Onboarding time** : -40% temps formation nouveaux utilisateurs
- **Future readiness** : Architecture prête multi-tenant

---

## Risques et mitigations

### Risques techniques

**Risque** : Performance dégradée par rapport à l'actuel  
**Mitigation** : Benchmarking continu, lazy loading, code splitting

**Risque** : Régression fonctionnelle lors migration  
**Mitigation** : Tests E2E complets, période coexistence

**Risque** : Courbe d'apprentissage équipe  
**Mitigation** : Formation Vue.js, documentation détaillée

### Risques utilisateur

**Risque** : Résistance au changement interface  
**Mitigation** : Migration progressive, formation, support utilisateur

**Risque** : Perte ergonomie actuelle  
**Mitigation** : Tests utilisateurs réguliers, préservation workflow

---

## Next steps immédiats

### Actions prioritaires (semaine 1)

1. **Validation stakeholders** : Approbation du plan et budget
2. **Setup équipe** : Recrutement/formation développeur Vue.js
3. **Prototype rapide** : Maquette interactive Figma
4. **Architecture PoC** : Setup Vite + Vue 3 + premier composant
5. **Planning détaillé** : Sprint planning 12 semaines

### Livrables semaine 1
- [ ] Plan validé et signé
- [ ] Équipe constituée et formée
- [ ] Environnement développement setup
- [ ] Premier prototype navigable
- [ ] Roadmap détaillée avec jalons

Cette refonte transformera RAGpy en une application web moderne, scalable et prête pour les défis futurs tout en préservant son excellente ergonomie actuelle.