#!/usr/bin/env sh
set -eu

cd /app

mkdir -p /app/runtime /app/runtime/sessions /app/user_matrices /app/cache

seed_cache_from_dir() {
  seed_dir="${SIMSEARCH_SEED_CACHE_DIR:-}"
  if [ -z "$seed_dir" ] || [ ! -d "$seed_dir" ]; then
    return 0
  fi

  echo "[entrypoint] Checking cache seed directory: ${seed_dir}"
  for metric in L2 IP; do
    src_dir="${seed_dir}/${metric}"
    dst_dir="/app/cache/${metric}"
    [ -d "$src_dir" ] || continue

    mkdir -p "$dst_dir"
    for src_file in "$src_dir"/*_ids.npy; do
      [ -f "$src_file" ] || continue
      dst_file="${dst_dir}/$(basename "$src_file")"
      if [ -s "$dst_file" ]; then
        echo "[entrypoint] Cache mapping already exists: ${dst_file}"
        continue
      fi
      echo "[entrypoint] Seeding cache mapping: ${src_file} -> ${dst_file}"
      cp -p "$src_file" "$dst_file"
    done
  done

  if [ ! -f "${SIMSEARCH_IMAGE_NAMES_FILE_L2:-/app/cache/L2/21000000_ids.npy}" ] && \
     [ ! -f "${SIMSEARCH_IMAGE_NAMES_FILE_IP:-/app/cache/IP/21000000_ids.npy}" ]; then
    echo "[entrypoint] No configured FAISS mapping file found after cache seeding."
  fi
}

case "${SIMSEARCH_SEED_CACHE_ON_START:-1}" in
  1|true|TRUE|yes|YES|on|ON)
    seed_cache_from_dir
    ;;
esac

case "${SIMSEARCH_CLEAR_USER_MATRICES_ON_START:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    echo "[entrypoint] Clearing /app/user_matrices contents..."
    PRESERVE_DIRS_RAW="${SIMSEARCH_PRESERVE_USER_MATRICES_DIRS:-}"
    PRESERVE_DIRS="$(printf '%s' "$PRESERVE_DIRS_RAW" | tr -d '[:space:]')"
    for entry in /app/user_matrices/* /app/user_matrices/.[!.]* /app/user_matrices/..?*; do
      [ -e "$entry" ] || continue
      base_name="$(basename "$entry")"
      case ",$PRESERVE_DIRS," in
        *,"$base_name",*)
          echo "[entrypoint] Preserving /app/user_matrices/$base_name"
          continue
          ;;
      esac
      rm -rf "$entry"
    done
    ;;
esac

echo "[entrypoint] Applying migrations..."
SIMSEARCH_SKIP_APP_AUTOLOAD=1 python manage.py migrate --noinput --skip-checks

probe_native_import() {
  module_name="$1"
  display_name="$2"
  required="$3"

  echo "[entrypoint] Probing ${display_name} import..."
  if python -c "import importlib; importlib.import_module('${module_name}')" >/tmp/simsearch-native-probe.err 2>&1; then
    echo "[entrypoint] ${display_name} import OK"
    return 0
  fi

  status=$?
  if grep -qi 'illegal instruction' /tmp/simsearch-native-probe.err 2>/dev/null; then
    status=132
  fi
  echo "[entrypoint] ${display_name} import failed with exit code ${status}"
  if [ "$status" -eq 132 ]; then
    echo "[entrypoint] ${display_name} hit an illegal CPU instruction (SIGILL). The installed wheel likely requires CPU instructions unavailable on this host."
  fi
  if [ -s /tmp/simsearch-native-probe.err ]; then
    sed 's/^/[entrypoint] probe stderr: /' /tmp/simsearch-native-probe.err
  fi

  if [ "$required" = "required" ]; then
    exit "$status"
  fi
  return 1
}

echo "[entrypoint] Probing native Python imports..."
probe_native_import numpy NumPy required
if ! probe_native_import faiss FAISS optional; then
  echo "[entrypoint] FAISS is unavailable; startup will report a dataset/index load error instead of crashing."
fi
if ! probe_native_import torch PyTorch optional; then
  echo "[entrypoint] PyTorch is unavailable; disabling CLIP model auto-load for this container start."
  export SIMSEARCH_AUTO_LOAD_CLIP_MODEL=0
fi

echo "[entrypoint] Starting gunicorn on 0.0.0.0:${PORT:-8942}"
PRELOAD_ARG=""
if [ "${GUNICORN_PRELOAD:-1}" = "1" ]; then
  PRELOAD_ARG="--preload"
fi
echo "[entrypoint] Gunicorn config: preload=${GUNICORN_PRELOAD:-1} class=${GUNICORN_WORKER_CLASS:-sync} workers=${GUNICORN_WORKERS:-2} threads=${GUNICORN_THREADS:-1} timeout=${GUNICORN_TIMEOUT:-0}"

exec gunicorn djangoProject.wsgi:application \
  --bind 0.0.0.0:${PORT:-8942} \
  ${PRELOAD_ARG} \
  --worker-class ${GUNICORN_WORKER_CLASS:-sync} \
  --workers ${GUNICORN_WORKERS:-2} \
  --threads ${GUNICORN_THREADS:-1} \
  --timeout ${GUNICORN_TIMEOUT:-0} \
  --graceful-timeout ${GUNICORN_GRACEFUL_TIMEOUT:-120} \
  --keep-alive ${GUNICORN_KEEPALIVE:-5} \
  --max-requests ${GUNICORN_MAX_REQUESTS:-0} \
  --max-requests-jitter ${GUNICORN_MAX_REQUESTS_JITTER:-0} \
  --worker-tmp-dir ${GUNICORN_WORKER_TMP_DIR:-/dev/shm} \
  --capture-output \
  --access-logfile - \
  --error-logfile -
