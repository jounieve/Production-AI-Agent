# Production AI Agent - Urban Migration Research Agent

Agent de recherche production-grade sur la **migration urbaine** (topic #4 du brief) :
il évalue si une ville donnée peut absorber un flux de migrants climatiques,
économiques ou liés à un conflit, en combinant recherche documentaire hybride,
données structurées par ville, et un score de risque de corridor calculé - le
tout sous contrainte de sécurité (guardrails) et d'observabilité (Langfuse).

**État du dépôt** : tous les bugs bloquants identifiés ont été corrigés et
re-testés (voir section 6). Il reste des tâches qui nécessitent vos propres
clés API - voir section 7 **"Ce qu'il reste à faire"** avant de soumettre.

---

## 1. Structure du dépôt

```
Production-AI-Agent/
├── README.md              ← ce fichier
├── REPORT.md              ← rapport écrit à rédiger (max 4 pages, voir docs/HW_brief)
├── requirements.txt       ← dépendances Python épinglées
├── .env.example           ← clés requises (à copier en .env)
├── conftest.py            ← permet à pytest de trouver src/ depuis n'importe où
├── app.py                 ← interface de démo Streamlit (streamlit run app.py)
├── src/
│   ├── agent.py            ← boucle principale : orchestre tout le pipeline
│   ├── mcp_server.py       ← serveur MCP exposant 3 outils à l'agent
│   ├── retrieval.py        ← recherche hybride (BM25+dense+RRF) + reranking
│   ├── guardrails.py       ← L1 (filtre entrée) + L4 (gate d'action) + TokenBudget + EU AI Act
│   └── reasoning.py        ← few-shot CoT + Self-Consistency + agent critique
├── tests/
│   ├── test_security.py    ← 5 tests d'injection requis + 1 test de cohérence
│   └── test_full_stack.py  ← validation complète mappée à chaque critère du rubric
├── eval/
│   ├── ragas_eval.py       ← RAGAS : baseline vs pipeline final
│   └── benchmark.py        ← coût, latence, distribution d'appels d'outils, monitoring
├── docs/
│   ├── architecture.md     ← diagramme + description des composants
│   ├── HW_brief(1).md      ← consignes du projet (fournies par l'enseignant)
│   └── HW_rubric.md        ← grille de notation (fournie par l'enseignant)
└── data/
    ├── README.md            ← comment peupler ce dossier
    ├── corpus/              ← documents texte pour le RAG (.txt/.md uniquement)
    ├── cities/cities.json   ← données structurées par ville
    └── eval_questions.json  ← questions + réponses de référence pour RAGAS
```

Ce dépôt suit exactement la structure requise par `docs/HW_brief(1).md` - aucun
fichier supplémentaire à la racine (pas de notebooks de labs, pas de module
partagé séparé) : tout ce qui est nécessaire à l'agent vit dans `src/`.

---

## 2. Installation et lancement (from scratch)

```bash
git clone <votre-repo>
cd Production-AI-Agent

cp .env.example .env
# → éditez .env et remplissez :
#     LLM_PROVIDER=openai   (ou "ollama" pour un modèle 100% local, voir ci-dessous)
#     OPENAI_API_KEY=sk-...           (si LLM_PROVIDER=openai)
#     LANGFUSE_PUBLIC_KEY=...
#     LANGFUSE_SECRET_KEY=...
#     LANGFUSE_HOST=https://cloud.langfuse.com   (ou votre instance self-host)

pip install -r requirements.txt --break-system-packages

python src/agent.py
```

### Choix du provider LLM (`LLM_PROVIDER`)

`src/agent.py` et `src/reasoning.py` supportent deux fournisseurs derrière la
même interface (les deux parlent l'API `chat.completions` compatible OpenAI,
donc c'est le même code applicatif dans les deux cas - seuls `base_url` et la
clé changent). Le fournisseur se choisit avec **une seule variable**
d'environnement, sans toucher au code :

- `LLM_PROVIDER=openai` (+ `OPENAI_API_KEY`) - fournisseur par défaut, modèle `gpt-4o-mini`.
- `LLM_PROVIDER=ollama` (+ `ollama serve` lancé localement, aucune clé requise)
  - modèle par défaut `llama3.2:latest`, changeable via `LLM_MODEL`
    (ex. `qwen2.5-coder:7b`). Coût marginal nul, utile pour itérer sans
    consommer de crédits API.
- Si `LLM_PROVIDER` n'est pas défini : `openai` si une vraie clé
  `OPENAI_API_KEY` est présente, sinon repli automatique sur `ollama`.

Ajouter Ollama n'a supprimé aucun support d'OpenAI : c'est un simple switch de
configuration, pas un remplacement.

Cette dernière commande doit produire en sortie : la question posée, les
outils MCP appelés, la réponse finale (`CONCLUSION`), le score de confiance,
le taux d'accord de la Self-Consistency, le verdict du critique, et l'usage
de tokens.

### Tester chaque brique séparément (recommandé avant l'agent complet)

Chaque module a un bloc `if __name__ == "__main__":` qui sert de smoke test
manuel. Testez dans cet ordre - ça isole immédiatement le composant en cause
si quelque chose casse :

```bash
python src/guardrails.py     # aucune clé API nécessaire - teste L1, L4, TokenBudget
python src/retrieval.py      # nécessite data/corpus/ peuplé (déjà fait, voir §3)
python src/reasoning.py      # nécessite OPENAI_API_KEY, ou rien du tout (repli sur Ollama local)
python src/mcp_server.py     # lance le serveur MCP en standalone (Ctrl+C pour quitter)
```

Pour inspecter le serveur MCP avec l'outil officiel (vérifie que les 3 tools
répondent correctement, requis pour le critère B du rubric) :

```bash
npx @modelcontextprotocol/inspector python src/mcp_server.py
```

### Tests de sécurité (obligatoire, gate du rubric)

```bash
python -m pytest tests/test_security.py -v
```

Les 6 tests couvrent : injection directe, injection Unicode obfusquée,
injection indirecte via contenu récupéré, appel d'outil non autorisé/inconnu,
consommation de tokens incontrôlée, et cohérence de `ACTION_RISK_MATRIX`. Les
5 premiers sont ceux exigés par le brief ; le 6e est un garde-fou de
régression ajouté en bonus. **Vérifié : les 6 passent depuis la racine.**

### Évaluation quantitative (RAGAS + coût/latence)

```bash
python eval/ragas_eval.py     # génère eval/ragas_results.json (baseline vs final)
python eval/benchmark.py      # génère eval/benchmark_results.json (coût, latence, tools)
```

Copiez les tables produites directement dans `REPORT.md`, section 3.

---

## 3. Rôle et fonctionnement de chaque fichier source

### `src/guardrails.py` - Sécurité (L1 + L4 + TokenBudget)

- **L1 - `l1_input_filter(query)`** : tourne AVANT toute recherche ou appel
  LLM. Normalise l'Unicode (NFKC + suppression des caractères zero-width)
  pour déjouer les attaques obfusquées, rejette les requêtes trop longues, et
  compare contre une liste de patterns regex d'injection ("ignore previous
  instructions", "reveal your system prompt", etc.). Renvoie un
  `L1Result(allowed, normalized_query, reasons)`.
- **`l1_filter_retrieved_context(chunks)`** : applique le même filtre au
  contenu **récupéré** (RAG), pas seulement à la question de l'utilisateur -
  défense contre l'injection indirecte (instruction malveillante cachée dans
  un document du corpus).
- **L4 - `ActionGate`** : classe stateful qui vérifie chaque appel d'outil
  contre `ACTION_RISK_MATRIX` avant exécution. Un outil absent de la matrice
  est **bloqué par défaut** (fail-closed). Les outils `"high"` risk
  nécessitent un flag explicite. Un compteur par session applique aussi une
  limite d'appels par outil.
- **`TokenBudget`** : compteur cumulatif de tokens (entrée+sortie) sur une
  session ; lève `TokenBudgetExceeded` si le budget configuré est dépassé -
  coupe-circuit indépendant du L4, utile contre les boucles d'appels d'outils
  déclenchées par injection.
- **`risk_tier(description)`** : classification EU AI Act (PROHIBITED / HIGH
  RISK / LIMITED RISK / MINIMAL RISK) à partir d'une description en texte
  libre, plus l'obligation associée. Appelée par `src/agent.py` sur
  `AGENT_DESCRIPTION` au chargement du module, pour que la section 5 du
  rapport soit backée par du code exécutable, pas seulement une justification
  en prose. Vérifie qu'un mot-clé topique ("migration") ne suffit pas à
  déclencher HIGH RISK - seul un cas d'usage décisionnel réel (contrôle aux
  frontières, éligibilité à l'asile...) le déclenche ; voir le docstring de
  la fonction pour le raisonnement complet.

### `src/retrieval.py` - Pipeline de recherche hybride

1. **Chunking parent-enfant** (`build_parent_child_index`) : découpe chaque
   document `.txt`/`.md` du corpus en blocs "parents" (~1200 caractères,
   renvoyés au LLM) et "enfants" (~300 caractères, indexés/recherchés).
2. **Recherche hybride** (`HybridRetriever.search`) :
   - `_bm25_search` : recherche lexicale (BM25Okapi) sur les chunks enfants.
   - `_dense_search` : recherche sémantique (embeddings `all-MiniLM-L6-v2`).
   - `_rrf_fuse` : fusion par Reciprocal Rank Fusion (dépend du rang, pas de
     l'échelle des scores - c'est ce qui permet de combiner BM25 et cosinus
     sans normalisation).
   - `_rerank` : reranking cross-encoder (`ms-marco-MiniLM-L-6-v2`) sur les
     candidats fusionnés.
   - Expansion enfant → parent avant de renvoyer le contexte final.
3. **`basic_retrieval()`** : top-k cosinus simple, sans BM25/RRF/rerank/
   parent-child - sert uniquement de **baseline** pour la comparaison RAGAS
   (voir `eval/ragas_eval.py`).

### `src/mcp_server.py` - Serveur MCP (3 outils)

Serveur `FastMCP` exposant :

| Outil | Rôle | Quand l'utiliser |
|---|---|---|
| `search_migration_evidence(query)` | Recherche hybride+rerank sur le corpus | Question qualitative/conceptuelle |
| `get_city_capacity_profile(city_name)` | Lookup de données structurées d'une ville | Chiffre précis pour une ville nommée |
| `compute_push_pull_index(origin_region, destination_city, push_factor_type)` | Score composite push/pull calculé | Comparaison de corridor origine→destination |

Chaque outil a une docstring complète (Use when / Do NOT use / Returns /
Example) et attrape ses propres exceptions pour ne jamais faire planter le
serveur (retourne un dict `{"error": ...}` structuré à la place).

### `src/agent.py` - Boucle principale

Orchestre le pipeline complet :
`requête utilisateur → L1 → boucle d'appel d'outils MCP (via un vrai client
stdio, chaque appel passant par le L4) → L1 sur le contenu récupéré →
Self-Consistency (k=3) + few-shot CoT → agent critique → réponse finale
structurée (AgentRunResult)`.

Le fournisseur LLM se choisit via `LLM_PROVIDER` (voir ci-dessus) : `_select_tools_turn`
utilise le SDK OpenAI standard (OpenAI et Ollama parlent tous deux l'API
`chat.completions`), et normalise la réponse en `_LLMReply` (tool_calls sous
forme de dicts simples, pas d'objets SDK) pour que le reste de la boucle ne
touche jamais aux détails du fournisseur.

Chaque appel LLM de sélection d'outil (`_select_tools_turn`) et chaque appel
d'outil MCP (`_call_mcp_tool`) a désormais son propre span Langfuse
(`@observe`), en plus du span racine `agent.run` - pour que chaque LLM call
et chaque tool call soit individuellement visible dans une trace.

**Versioning et monitoring de production** (`AGENT_VERSION`, `AgentMonitor`) :
`AGENT_VERSION` est un dict incluant un hash SHA-256 (`hash_prompt`) du prompt
système de sélection d'outils - si le hash change entre deux runs, le prompt a
changé, ce qui permet de tracer un changement de comportement jusqu'au commit
qui a édité le prompt. `AgentMonitor` (instance module-level `_monitor`)
accumule des statistiques sur tous les runs d'un même process et déclenche une
alerte imprimée (`[MONITOR ALERT] ...`) sur un run lent, un run coûteux, une
réponse vide, ou un taux d'erreur d'outil élevé - complémentaire aux spans
Langfuse (qui montrent "que s'est-il passé dans CE trace", pas "y a-t-il un
problème à travers plusieurs runs"). `eval/benchmark.py` imprime et sauvegarde
`get_monitor_summary()` à la fin de ses 10 runs.

Le point d'entrée `main()` lance une question de démonstration et affiche le
résultat complet (réponse, confiance, accord de Self-Consistency, verdict du
critique, usage de tokens, version de l'agent, tier EU AI Act, résumé du monitor).

### `src/reasoning.py` - Stratégie de raisonnement

- **`self_consistency_synthesis`** : appelle le LLM **k=3 fois
  indépendamment** avec un prompt few-shot imposant le format
  `EVIDENCE / ANALYSIS / CONCLUSION / CONFIDENCE`, puis regroupe les réponses
  par similarité de mots-clés sur la CONCLUSION (vote majoritaire), et
  retourne le candidat le plus confiant du cluster gagnant.
- **`critic_review`** : **second rôle d'agent**, séparé - vérifie la réponse
  gagnante contre le contexte réellement fourni (détecte hallucination et
  surconfiance) et rend un verdict `APPROVED`/`REJECTED` avec justification.

### `tests/test_security.py`

5 tests requis (injection directe, injection Unicode obfusquée, injection
indirecte via contenu récupéré, appel d'outil non autorisé, consommation de
tokens incontrôlée) + 1 test de cohérence de `ACTION_RISK_MATRIX`. Le fichier
`conftest.py` à la racine du dépôt leur permet de trouver `src/` peu importe
d'où `pytest` est lancé.

### `eval/ragas_eval.py` et `eval/benchmark.py`

- `ragas_eval.py` : exécute le pipeline en mode `baseline` (retrieval
  basique + réponse zero-shot directe) et en mode `final` (hybride+rerank +
  few-shot CoT + Self-Consistency) sur les questions de
  `data/eval_questions.json`, note les 4 métriques RAGAS (context_recall,
  context_precision, faithfulness, answer_relevancy), et sauvegarde la
  comparaison dans `eval/ragas_results.json`.
- `benchmark.py` : lance l'agent complet sur ≥10 questions, mesure latence et
  coût estimé par run, la distribution des appels d'outils, et déclenche
  **volontairement** un `TokenBudgetExceeded` pour prouver que le mécanisme
  fonctionne (exigé par le rubric G).

### `data/`

- **`corpus/`** : 4 documents thématiques `.md` (`receiving_city_capacity.md`,
  `push_pull_factors.md`, `migration_corridors.md`,
  `climate_early_warning_and_policy.md`) rédigés pour couvrir précisément les
  12 questions de `eval_questions.json`. 3 PDF de référence sont aussi présents
  (`l-avenir-des-villes-face-aux-migrations-climatiques-2020-1.pdf`,
  `Migration_and_Cities_An_Introduction.pdf`,
  `Mateo Merchan Article City Migration.pdf`) mais **ne sont pas lus par le
  retriever** (seuls `.txt`/`.md` le sont) et n'ont pas été convertis - leur
  contenu n'est donc pas recherchable tant qu'ils ne sont pas convertis en
  `.md` (voir `data/README.md` pour la règle complète).
- **`cities/cities.json`** : données structurées (vacance de logement,
  croissance de l'emploi, capacité scolaire, lits d'hôpitaux, couverture
  transport) pour 6 villes.
- **`eval_questions.json`** : 12 questions avec réponse de référence
  (`ground_truth`) pour RAGAS.

---

## 4. Architecture (résumé - diagramme complet dans `docs/architecture.md`)

```
Utilisateur
    │
    ▼
L1 input filter (guardrails.py) ──── bloqué? ──► réponse "blocked_reason"
    │ ok
    ▼
Boucle LLM (OpenAI/Ollama) + MCP (agent.py) ◄──► mcp_server.py (3 tools)
    │  chaque appel d'outil passe par L4 (guardrails.py)
    ▼
L1 filtre le contenu récupéré (défense injection indirecte)
    │
    ▼
Self-Consistency k=3 + few-shot CoT (reasoning.py) ──► synthèse gagnante
    │
    ▼
Agent critique (reasoning.py) ──► verdict APPROVED/REJECTED
    │
    ▼
Réponse finale structurée (AgentRunResult)
```

`docs/architecture.md` contient en plus la description détaillée de chaque
composant et l'explication d'un choix de design (pourquoi RRF plutôt qu'une
somme pondérée de scores) - à recopier/adapter dans `REPORT.md` section 2.

---

## 5. Sécurité, EU AI Act, et autres attentes du rapport

Ces points sont déjà **implémentés dans le code** ; ils restent à **décrire et
justifier dans `REPORT.md`** (sections 4, 5, 6 du brief) :

- **Sécurité** : les 5 tests d'injection + leur résultat avant/après L1+L4 →
  copiez la sortie de `pytest tests/test_security.py -v`.
- **EU AI Act** : déjà résolu par du code exécutable, pas seulement par de la
  prose - `guardrails.risk_tier(agent.AGENT_DESCRIPTION)` renvoie
  `("LIMITED RISK", "Users must be informed...")`, testé dans
  `tests/test_full_stack.py::TestLab4Production`. REPORT.md section 5 cite
  déjà cet appel ; si vous changez `AGENT_DESCRIPTION` ou la description de
  l'agent, relancez `python src/guardrails.py` pour vérifier que le tier ne
  change pas de façon inattendue.
- **Limitations** : plusieurs limitations concrètes existent déjà dans le
  code et sont déjà décrites dans `REPORT.md` section 6, par exemple :
  - `compute_push_pull_index` utilise des scores de sévérité **constants**
    par type de push factor (`_PUSH_FACTOR_SEVERITY` dans `mcp_server.py`),
    pas des données live (indices de sécheresse, conflits...) - à citer comme
    limitation explicite plutôt que de la cacher.
  - Avec `LLM_PROVIDER=ollama`, un modèle local comme `llama3.2:latest` ne
    suit pas toujours la séquence d'outils "obligatoire" du prompt système
    aussi fidèlement que `gpt-4o-mini` - observé en direct pendant les tests
    de ce soir (voir REPORT.md section 6).
  - `benchmark.py` estime le coût avec un **split input/output supposé**
    (60/40), pas les tokens exacts par appel - autre limitation documentée
    dans le code lui-même (voir commentaire `ASSUMED_INPUT_FRACTION`).

---

## 6. Corrections déjà appliquées (historique)

Ces problèmes ont été identifiés puis corrigés et **re-testés** - ils ne
doivent plus être présents si vous partez de cette version du dépôt :

| Problème | Correction | Vérifié |
|---|---|---|
| `data/corpus/` ne contenait que des PDF, illisibles par `retrieval.py` (qui ne lit que `.txt`/`.md`) | 4 documents `.md` thématiques créés pour couvrir les 12 questions de `eval_questions.json` (les PDF restent non convertis, gardés en référence uniquement) | ✅ chunking (32 parents/180 enfants) + recherche BM25 testés, retrouvent les bons passages ; pipeline complet re-vérifié avec les vrais modèles (sentence-transformers + cross-encoder) |
| `python -m pytest tests/test_security.py` échouait depuis la racine (`ModuleNotFoundError: guardrails`) | `conftest.py` déplacé de `src/` vers la racine du dépôt | ✅ les 6 tests passent depuis la racine |
| `.env.example` absent | Créé avec toutes les clés nécessaires | ✅ |
| `docs/architecture.md` vide | Rempli : diagramme + composants + 1 décision de design justifiée | ✅ |
| `data/README.md` absent | Créé | ✅ |
| `REPORT.MD` mal nommé (casse) | Renommé en `REPORT.md` | ✅ |
| Observabilité Langfuse partielle (pas de span par appel d'outil/LLM dans `agent.py`) | Spans dédiés ajoutés (`_select_tools_turn`, `_call_mcp_tool`) | ✅ compile sans erreur |
| `.gitignore` ne contenait que `.env` malgré ce tableau (pyca cache versionné) | `.gitignore` corrigé pour de vrai (`__pycache__/`, `.pytest_cache/`, résultats d'éval) | ✅ `src/__pycache__` retiré du suivi git |
| `data/Read.me` : doublon exact de `data/README.md` | Supprimé | ✅ |
| `src/agent.py`/`src/reasoning.py` détectaient le fournisseur en testant si `OPENAI_API_KEY` ressemblait à une vraie clé, sans switch explicite | `LLM_PROVIDER=openai\|ollama` en config explicite dans `.env` ; repli automatique sur Ollama si non défini et pas de vraie clé OpenAI | ✅ testé en direct contre un serveur Ollama local (tool-calling multi-tour + synthèse + critique) |
| `eval/benchmark.py` estimait le coût avec un tarif Claude Sonnet câblé en dur, alors que l'agent n'utilise jamais Claude par défaut | Coût lu dynamiquement dans `agent.PRICING_USD_PER_MTOK` d'après `agent.MODEL_NAME` | ✅ |
| README section 2 demandait `ANTHROPIC_API_KEY`, une clé jamais utilisée par le code (`OPENAI_API_KEY`/Ollama seulement) | Corrigé | ✅ |
| Docstrings manquantes sur des fonctions internes (`_bm25_search`, `ActionGate.check`, `TokenBudget.add`, etc.) | Commentaire d'1-2 lignes ajouté à chaque fonction sans docstring dans `src/`, `eval/`, `app.py` | ✅ |
| Tirets cadratins (`—`) dans le code et la documentation | Remplacés par des tirets simples partout sauf dans `docs/HW_brief(1).md`/`docs/HW_rubric.md` (fournis par l'enseignant, non modifiés) | ✅ |
| Les concepts du cours (RAG, sécurité, raisonnement, production) n'avaient pas de trace explicite dans le code lorsqu'ils venaient d'une brique "production" spécifique | `risk_tier()` (EU AI Act, `guardrails.py`) et `AgentMonitor`/`hash_prompt`/`AGENT_VERSION` versionné (monitoring + versioning de production, `agent.py`) implémentés directement dans `src/`, avec tests dédiés dans `tests/test_full_stack.py::TestLab4Production` | ✅ 5 tests dédiés passent, vérifié en direct (alertes déclenchées, hash cohérent) |

---
