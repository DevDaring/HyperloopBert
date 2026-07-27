#!/usr/bin/env bash
# autopush.sh -- push code + results to the private GitHub repo on an interval.
#
# SECURITY CONTRACT (do not weaken):
#   * The GitHub token is read at runtime from Codes/.env (var Github_Classic_Token)
#     or the environment. It is NEVER written to .git/config, a remote URL, the
#     process argument list, or any log.
#   * .env and secret-like files are NEVER pushed (gitignore + an explicit staged
#     -file guard that aborts the commit if anything secret slips through).
#   * results/ and figures/ are gitignored for local hygiene, so they are
#     force-added here on purpose; data/, models/, checkpoints/ stay ignored
#     (too large, not requested).
#
# Usage:
#   ./autopush.sh            # loop forever, push every $INTERVAL seconds
#   ./autopush.sh --once     # single push (for cron)
#   INTERVAL=900 BRANCH=main ./autopush.sh
set -euo pipefail            # NOTE: never `set -x` here (would echo the token)

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

INTERVAL="${INTERVAL:-900}"                 # 15 minutes
BRANCH="${BRANCH:-main}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/Codes/.env}"
LOG_FILE="${AUTOPUSH_LOG:-$REPO_DIR/autopush.log}"   # gitignored (*.log)

log() { printf '%s  %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$LOG_FILE" >&2; }

# --- Resolve the token WITHOUT printing it -----------------------------------
# Precedence: existing env var, then Codes/.env. Only the value is captured into
# a shell variable; it is never echoed.
resolve_token() {
  if [ -n "${GITHUB_CLASSIC_TOKEN:-}" ]; then
    printf '%s' "$GITHUB_CLASSIC_TOKEN"; return 0
  fi
  if [ -f "$ENV_FILE" ]; then
    # grep the single line; strip key=, surrounding quotes and CR.
    local v
    v="$(grep -iE '^[[:space:]]*Github_Classic_Token[[:space:]]*=' "$ENV_FILE" | head -1 || true)"
    v="${v#*=}"
    v="$(printf '%s' "$v" | tr -d '"'"'"'\r' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    printf '%s' "$v"; return 0
  fi
  return 1
}

GH_PUSH_TOKEN="$(resolve_token || true)"
if [ -z "${GH_PUSH_TOKEN:-}" ]; then
  log "ERROR: no GitHub token found (env GITHUB_CLASSIC_TOKEN or $ENV_FILE:Github_Classic_Token). Aborting."
  exit 1
fi
export GH_PUSH_TOKEN

# Credential helper reads the token from the exported env var at call time, so
# the literal token never lands in argv, .git/config, or logs.
CRED_HELPER='!f() { echo "username=x-access-token"; echo "password=${GH_PUSH_TOKEN}"; }; f'

# Ensure the remote is tokenless (defensive: scrub any embedded token).
git remote set-url origin https://github.com/DevDaring/HyperloopBert.git 2>/dev/null || true

# Commit identity (only set if missing).
git config user.name  >/dev/null 2>&1 || git config user.name  "HyperloopBERT Autopush"
git config user.email >/dev/null 2>&1 || git config user.email "koushikdeb2009@gmail.com"

# --- Secret guard: refuse to commit anything that looks like a secret --------
SECRET_RE='(^|/)\.env(\.|$)|\.pem$|\.key$|(^|/)secrets?/|id_rsa|credentials?\.json|\.pgpass|\.netrc|\.p12$'

push_once() {
  # Stage code (gitignore keeps data/models/checkpoints/.env out), then FORCE
  # results + figures which are gitignored for local hygiene. dry_run/ output
  # (Dry_Run harness sandbox) is deliberately EXCLUDED: it's QA scaffolding,
  # not experiment results, and must never mix into the real results history.
  git add -A
  for base in Codes/results Codes/figures results figures; do
    [ -d "$base" ] || continue
    for sub in "$base"/*/; do
      name="$(basename "$sub")"
      [ "$name" = "dry_run" ] && continue
      git add -f "$sub" 2>/dev/null || true
    done
  done

  # Abort if any secret-like path is staged.
  local staged
  staged="$(git diff --cached --name-only || true)"
  if printf '%s\n' "$staged" | grep -qiE "$SECRET_RE"; then
    log "ABORT: a secret-like file is staged; unstaging and skipping this push."
    printf '%s\n' "$staged" | grep -iE "$SECRET_RE" | sed 's/^/  blocked: /' | tee -a "$LOG_FILE" >&2
    git reset -q
    return 1
  fi

  if git diff --cached --quiet; then
    log "no changes to push."
    return 0
  fi

  local n
  n="$(git diff --cached --name-only | wc -l | tr -d ' ')"
  git commit -q -m "auto: results/code sync $(date -u +%Y-%m-%dT%H:%M:%SZ) (${n} files)"
  # Rebase on remote to avoid diverging history, then push. Credentials come
  # from the in-memory helper; the token is never in the command line.
  git -c credential.helper="$CRED_HELPER" pull --rebase --autostash origin "$BRANCH" >>"$LOG_FILE" 2>&1 || \
    log "warn: pull --rebase failed (continuing to push)."
  if git -c credential.helper="$CRED_HELPER" push origin "HEAD:$BRANCH" >>"$LOG_FILE" 2>&1; then
    log "pushed ${n} changed file(s) to origin/$BRANCH."
  else
    log "ERROR: push failed (see $LOG_FILE)."
    return 1
  fi
}

if [ "${1:-}" = "--once" ]; then
  push_once || true
  exit 0
fi

log "autopush started: every ${INTERVAL}s to origin/$BRANCH (Ctrl-C to stop)."
while true; do
  push_once || true
  sleep "$INTERVAL"
done
