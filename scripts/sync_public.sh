#!/bin/bash
# sync_public.sh — Synchronise axiom-public depuis axiom-private
#
# Synchronise le code source AXIOM (engine, abstraction, ai, solver, training,
# screen, scripts, config) + docs/ + tests/ + fichiers racine, pour permettre
# à un Claude consultant (claude.ai) de lire le code via web_fetch.
#
# Exclut : data/ (poids), .git/, __pycache__, *.pkl/*.pt/*.npz (modèles),
#          .env*, *.secrets, credentials*, venv/, .vscode/, .idea/, *.log, *.zip
#
# Usage : bash scripts/sync_public.sh
#         bash scripts/sync_public.sh --dry-run    # aperçu sans copier
#
# Appelé automatiquement par le hook post-commit si modifs détectées.

set -e

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    echo "=== MODE DRY-RUN — aucune modification ==="
fi

PRIVATE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PUBLIC_DIR="$(cd "$PRIVATE_DIR/../axiom-public" 2>/dev/null && pwd)" || {
    echo "✗ axiom-public introuvable à $PRIVATE_DIR/../axiom-public"
    exit 1
}

echo "→ Sync depuis : $PRIVATE_DIR"
echo "→ Vers        : $PUBLIC_DIR"

# ─── Items à synchroniser (dossiers + fichiers racine) ────────────────────────
SYNC_ITEMS=(
    # Code source
    "engine"
    "abstraction"
    "ai"
    "solver"
    "training"
    # "screen" — exclu (scraper Betclic, pas utile au consultant Claude)
    "scripts"
    "config"
    # Documentation et tests
    "docs"
    "tests"
    # Fichiers racine
    "CLAUDE.md"
    "requirements.txt"
    "README.txt"
    "TODO.txt"
    "ARCHITECTURE.txt"
    "PLAN_ACTION.txt"
    "libratus.txt"
    "pytest.ini"
    "train.py"
    "train_hu.py"
    "main.py"
)

# ─── Patterns d'exclusion à pruner après copie ────────────────────────────────
EXCLUDE_DIRS=(
    "__pycache__"
    ".pytest_cache"
    "venv"
    ".venv"
    "env"
    ".vscode"
    ".idea"
    "node_modules"
)

EXCLUDE_FILES=(
    "*.pyc"
    "*.pyo"
    "*.pkl"
    "*.pt"
    "*.npz"
    "*.log"
    "*.zip"
    ".env"
    ".env.*"
    "*.secrets"
    "credentials*.json"
    "*.pem"
    "*.key"
    "*.swp"
    "*.swo"
    ".DS_Store"
    "Thumbs.db"
)

# ─── 1. Suppression des items existants côté public (sauf .git/) ──────────────
if [ "$DRY_RUN" = "0" ]; then
    for ITEM in "${SYNC_ITEMS[@]}"; do
        DST="$PUBLIC_DIR/$ITEM"
        [ -e "$DST" ] && rm -rf "$DST"
    done
fi

# ─── 2. Copie des items ───────────────────────────────────────────────────────
echo ""
echo "→ Copie des items :"
for ITEM in "${SYNC_ITEMS[@]}"; do
    SRC="$PRIVATE_DIR/$ITEM"
    DST="$PUBLIC_DIR/$ITEM"
    if [ -e "$SRC" ]; then
        if [ "$DRY_RUN" = "1" ]; then
            if [ -d "$SRC" ]; then
                COUNT=$(find "$SRC" -type f \
                    -not -path "*/__pycache__/*" -not -path "*/.pytest_cache/*" \
                    -not -name "*.pyc" -not -name "*.pyo" \
                    -not -name "*.pkl" -not -name "*.pt" -not -name "*.npz" \
                    -not -name "*.log" -not -name "*.zip" 2>/dev/null | wc -l)
                echo "  [DRY-RUN] $ITEM/  ($COUNT fichiers)"
            else
                echo "  [DRY-RUN] $ITEM"
            fi
        else
            if [ -d "$SRC" ]; then
                cp -r "$SRC" "$DST"
            else
                cp "$SRC" "$DST"
            fi
            echo "  ✓ $ITEM"
        fi
    else
        echo "  - $ITEM (absent, ignoré)"
    fi
done

# ─── 3. Prune des exclusions côté public ──────────────────────────────────────
if [ "$DRY_RUN" = "0" ]; then
    echo ""
    echo "→ Prune exclusions :"
    for D in "${EXCLUDE_DIRS[@]}"; do
        find "$PUBLIC_DIR" -type d -name "$D" -not -path "*/.git/*" -prune -exec rm -rf {} + 2>/dev/null || true
    done
    for F in "${EXCLUDE_FILES[@]}"; do
        find "$PUBLIC_DIR" -type f -name "$F" -not -path "*/.git/*" -delete 2>/dev/null || true
    done
    echo "  ✓ Exclusions appliquées"
fi

# ─── 4. Récap dry-run + aperçu fichiers ───────────────────────────────────────
if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "→ Aperçu des 30 premiers fichiers source qui seraient synchronisés :"
    cd "$PRIVATE_DIR"
    for ITEM in "${SYNC_ITEMS[@]}"; do
        if [ -d "$ITEM" ]; then
            find "$ITEM" -type f \
                -not -path "*/__pycache__/*" -not -path "*/.pytest_cache/*" \
                -not -name "*.pyc" -not -name "*.pyo" \
                -not -name "*.pkl" -not -name "*.pt" -not -name "*.npz" \
                -not -name "*.log" -not -name "*.zip" 2>/dev/null
        elif [ -f "$ITEM" ]; then
            echo "$ITEM"
        fi
    done | head -30

    echo ""
    echo "→ Total fichiers qui seraient synchronisés :"
    TOTAL=0
    for ITEM in "${SYNC_ITEMS[@]}"; do
        if [ -d "$ITEM" ]; then
            N=$(find "$ITEM" -type f \
                -not -path "*/__pycache__/*" -not -path "*/.pytest_cache/*" \
                -not -name "*.pyc" -not -name "*.pyo" \
                -not -name "*.pkl" -not -name "*.pt" -not -name "*.npz" \
                -not -name "*.log" -not -name "*.zip" 2>/dev/null | wc -l)
            TOTAL=$((TOTAL + N))
        elif [ -f "$ITEM" ]; then
            TOTAL=$((TOTAL + 1))
        fi
    done
    echo "  $TOTAL fichiers"

    echo ""
    echo "→ Sanity check exclusions (les patterns suivants ne doivent JAMAIS apparaître ci-dessus) :"
    SUSPECT=0
    for ITEM in "${SYNC_ITEMS[@]}"; do
        if [ -d "$ITEM" ]; then
            BAD=$(find "$ITEM" -type f \( -name "*.pkl" -o -name "*.pt" -o -name "*.npz" -o -name "*.log" -o -name ".env*" -o -name "*.secrets" \) 2>/dev/null | head -3)
            if [ -n "$BAD" ]; then
                echo "  ⚠️  $BAD"
                SUSPECT=$((SUSPECT + 1))
            fi
        fi
    done
    [ "$SUSPECT" = "0" ] && echo "  ✓ Aucun fichier sensible détecté dans les SYNC_ITEMS"

    echo ""
    echo "=== DRY-RUN terminé — relancer sans --dry-run pour appliquer ==="
    exit 0
fi

# ─── 5. Commit + push dans axiom-public ───────────────────────────────────────
cd "$PUBLIC_DIR"

if ! git diff --quiet || ! git diff --cached --quiet || [ -n "$(git ls-files --others --exclude-standard)" ]; then
    DATE=$(date +%Y-%m-%d)
    LAST_COMMIT=$(cd "$PRIVATE_DIR" && git log -1 --pretty=%h)
    LAST_MSG=$(cd "$PRIVATE_DIR" && git log -1 --pretty=%s)
    git add -A
    git commit -m "sync: $DATE — $LAST_COMMIT $LAST_MSG"
    git push
    echo "✓ axiom-public mis à jour et pushé"
else
    echo "→ Aucun changement dans axiom-public, rien à committer"
fi
