#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${FASTKOKORO_VOICE_ENCODER_DATA:-$ROOT_DIR/data/public}"

mkdir -p "$DATA_DIR"

if [[ ! -d "$DATA_DIR/koniwa" ]]; then
  git clone https://github.com/koniwa/koniwa "$DATA_DIR/koniwa"
fi

cat <<EOF
Prepared:
  Koniwa: $DATA_DIR/koniwa

Manual download still required:
  SIWIS: https://datashare.ed.ac.uk/handle/10283/2353

Place SIWIS files under:
  $DATA_DIR/siwis
EOF
