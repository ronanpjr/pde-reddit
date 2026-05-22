#!/usr/bin/env bash
# download_data.sh
#
# Downloads Reddit Memes Part 1 and Part 2 from OSF, then extracts them into
# the correct directory structure expected by the OCR pipeline:
#
#   data/images/<subreddit>/   ← one directory per subreddit
#
# Each top-level zip contains per-subreddit sub-zips which are also extracted.
# Already-extracted subreddit directories are skipped on re-runs.
#
# Usage (run from the project root on the GPU server):
#   bash src/download_data.sh
#
# Requirements: wget, unzip (both standard on most Linux systems)

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGES_DIR="$PROJECT_ROOT/data/images"
TMP_DIR="$PROJECT_ROOT/data/_tmp_downloads"

PART1_URL="https://osf.io/download/f6eyx/"
PART2_URL="https://osf.io/download/emz63/"
PART1_ZIP="$TMP_DIR/Reddit_Memes_Part1.zip"
PART2_ZIP="$TMP_DIR/Reddit_Memes_Part2.zip"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { echo; log "▶ $*"; }
ok()   { log "✓ $*"; }
warn() { log "⚠ $*"; }

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
info "Setting up directories"
mkdir -p "$IMAGES_DIR" "$TMP_DIR"
ok "images → $IMAGES_DIR"
ok "tmp    → $TMP_DIR"

# ---------------------------------------------------------------------------
# Download function (skips if file already present and non-empty)
# ---------------------------------------------------------------------------
download() {
    local url="$1"
    local dest="$2"
    local label="$3"

    if [[ -f "$dest" && -s "$dest" ]]; then
        warn "$label already downloaded, skipping (delete $dest to re-download)"
        return
    fi

    info "Downloading $label (~$(numfmt --to=iec-i --suffix=B ${4:-0} 2>/dev/null || echo '?'))"
    wget \
        --no-verbose \
        --show-progress \
        --content-disposition \
        -O "$dest" \
        "$url"
    ok "$label → $dest"
}

# ---------------------------------------------------------------------------
# Extract a top-level part zip, then extract each inner subreddit zip
# ---------------------------------------------------------------------------
extract_part() {
    local zip_path="$1"
    local part_label="$2"

    info "Extracting $part_label: $zip_path"

    local part_tmp="$TMP_DIR/${part_label}_extracted"
    mkdir -p "$part_tmp"

    # Extract the top-level zip into a staging directory
    unzip -q -o "$zip_path" -d "$part_tmp"
    ok "Top-level zip extracted to $part_tmp"

    # Walk every .zip found inside and extract into data/images/
    local found=0
    while IFS= read -r -d '' sub_zip; do
        found=$((found + 1))
        local sub_name
        sub_name="$(basename "$sub_zip" .zip)"

        local dest_dir="$IMAGES_DIR/$sub_name"
        if [[ -d "$dest_dir" ]]; then
            local existing_count
            existing_count="$(find "$dest_dir" -maxdepth 1 -type f | wc -l)"
            if [[ "$existing_count" -gt 0 ]]; then
                warn "Skipping $sub_name ($existing_count files already present)"
                continue
            fi
        fi

        log "  Extracting subreddit: $sub_name"
        mkdir -p "$dest_dir"
        unzip -q -o "$sub_zip" -d "$dest_dir"
        local n
        n="$(find "$dest_dir" -maxdepth 1 -type f | wc -l)"
        ok "  $sub_name → $n files"
    done < <(find "$part_tmp" -name "*.zip" -print0)

    if [[ "$found" -eq 0 ]]; then
        # No inner zips: images may be directly inside the extracted folder
        warn "No inner .zip files found in $part_label — copying images directly"
        find "$part_tmp" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
            | while IFS= read -r img; do
                # Use parent directory name as subreddit
                local parent
                parent="$(basename "$(dirname "$img")")"
                local dest_dir="$IMAGES_DIR/$parent"
                mkdir -p "$dest_dir"
                cp "$img" "$dest_dir/"
            done
        ok "Direct images copied"
    fi
}

# ---------------------------------------------------------------------------
# Download both parts
# ---------------------------------------------------------------------------
download "$PART1_URL" "$PART1_ZIP" "Reddit Memes Part 1" 3665444773
download "$PART2_URL" "$PART2_ZIP" "Reddit Memes Part 2" 2987993822

# ---------------------------------------------------------------------------
# Extract both parts
# ---------------------------------------------------------------------------
extract_part "$PART1_ZIP" "part1"
extract_part "$PART2_ZIP" "part2"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
info "Done. Final image counts per subreddit:"
total=0
while IFS= read -r -d '' subdir; do
    n="$(find "$subdir" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)"
    total=$((total + n))
    printf "  %-45s %5d images\n" "$(basename "$subdir")" "$n"
done < <(find "$IMAGES_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo
log "Total images: $total"
log "Tmp files kept at: $TMP_DIR  (delete manually to free space)"
log "You can now run: docker compose up -d --build"
