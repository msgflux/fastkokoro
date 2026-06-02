#!/usr/bin/env bash
# Create a release PR for fastkokoro.
# Usage: ./scripts/release.sh <version>
# Example: ./scripts/release.sh 0.2.0
# Example: ./scripts/release.sh 0.2.0rc1

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${RED}Error: version number required${NC}"
    echo "Usage: $0 <version>"
    exit 1
fi

NEW_VERSION="$1"

if [[ ! "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]]; then
    echo -e "${RED}Error: invalid version format${NC}"
    echo "Version must be X.Y.Z or X.Y.ZaN / X.Y.ZbN / X.Y.ZrcN"
    exit 1
fi

extract_repo_slug() {
    local remote_url="$1"
    echo "$remote_url" | sed -E 's#(git@github.com:|https://github.com/)##; s#\.git$##'
}

BASE_REMOTE="origin"
if git remote get-url upstream >/dev/null 2>&1; then
    BASE_REMOTE="upstream"
fi

ORIGIN_REPO=$(extract_repo_slug "$(git remote get-url origin)")
BASE_REPO=$(extract_repo_slug "$(git remote get-url "$BASE_REMOTE")")
ORIGIN_OWNER="${ORIGIN_REPO%%/*}"

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}Error: must be on main branch${NC}"
    echo "Current branch: $CURRENT_BRANCH"
    exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
    echo -e "${RED}Error: working directory is not clean${NC}"
    git status --short
    exit 1
fi

echo -e "${BLUE}Pulling latest changes from ${BASE_REMOTE}/main...${NC}"
git pull --ff-only "$BASE_REMOTE" main

CURRENT_VERSION=$(python - <<'PY'
import tomllib
with open("pyproject.toml", "rb") as file:
    print(tomllib.load(file)["project"]["version"])
PY
)

uv run python - <<PY
from packaging.version import parse as parse_version
import sys

current = "$CURRENT_VERSION"
new = "$NEW_VERSION"

if parse_version(new) <= parse_version(current):
    print(f"Error: new version ({new}) must be greater than current version ({current})")
    sys.exit(1)

print(f"Version bump validated: {current} -> {new}")
PY

echo -e "${YELLOW}Releasing version: $CURRENT_VERSION -> $NEW_VERSION${NC}"

sed -i "0,/^version = \".*\"/s//version = \"$NEW_VERSION\"/" pyproject.toml

TODAY=$(date +%Y-%m-%d)
if grep -q "^## \\[Unreleased\\]" CHANGELOG.md; then
    sed -i "/## \[Unreleased\]/a \\
\\
## [$NEW_VERSION] - $TODAY" CHANGELOG.md
elif grep -q "^# Changelog" CHANGELOG.md; then
    awk -v version="$NEW_VERSION" -v today="$TODAY" '
        NR == 1 && $0 ~ /^# Changelog$/ {
            print $0
            print ""
            print "## [" version "] - " today
            next
        }
        { print }
    ' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md
else
    cat > CHANGELOG.md <<EOF
# Changelog

## [$NEW_VERSION] - $TODAY
EOF
fi

CHANGED_FILES=$(git diff --name-only)
UNEXPECTED_FILES=""
while IFS= read -r file; do
    [ -z "$file" ] && continue
    if [[ "$file" != "pyproject.toml" ]] && [[ "$file" != "CHANGELOG.md" ]]; then
        UNEXPECTED_FILES="${UNEXPECTED_FILES}${file}\n"
    fi
done <<< "$CHANGED_FILES"

if [ -n "$UNEXPECTED_FILES" ]; then
    echo -e "${RED}Unexpected files were modified:${NC}"
    echo -e "$UNEXPECTED_FILES"
    git checkout -- pyproject.toml CHANGELOG.md
    exit 1
fi

if ! grep -q "pyproject.toml" <<< "$CHANGED_FILES" || ! grep -q "CHANGELOG.md" <<< "$CHANGED_FILES"; then
    echo -e "${RED}Error: pyproject.toml and CHANGELOG.md should both be updated${NC}"
    exit 1
fi

BRANCH_NAME="release/v$NEW_VERSION"
git checkout -b "$BRANCH_NAME"

git add pyproject.toml CHANGELOG.md
git commit -m "RELEASE: v$NEW_VERSION" -m "Update fastkokoro from $CURRENT_VERSION to $NEW_VERSION."

git push -u origin "$BRANCH_NAME"

PR_HEAD="$BRANCH_NAME"
if [ "$ORIGIN_REPO" != "$BASE_REPO" ]; then
  PR_HEAD="${ORIGIN_OWNER}:${BRANCH_NAME}"
fi

PR_URL=$(gh pr create \
  --repo "$BASE_REPO" \
  --base main \
  --head "$PR_HEAD" \
  --title "RELEASE: v$NEW_VERSION" \
  --body "## Release v$NEW_VERSION

### Changes
- Version: $CURRENT_VERSION -> $NEW_VERSION
- Files modified: \`pyproject.toml\`, \`CHANGELOG.md\`

### After merge
1. \`Validate Release\` validates the release commit.
2. \`Publish Python Package\` builds, tags, publishes to PyPI, and creates the GitHub release.
3. \`Publish Docker Images\` publishes CPU and GPU images to Docker Hub.")

git checkout main

echo -e "${GREEN}Release PR created:${NC} $PR_URL"
