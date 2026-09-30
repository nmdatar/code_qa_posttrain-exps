#!/bin/sh
# Build and serve the local action-trace demo, retaining saved investigations.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
demo_url='http://127.0.0.1:8000/?mode=compare'
if [ "${1:-}" = '--open' ]; then
  demo_open=true
else
  demo_open=false
fi
# Reopening the demo should reuse its running backend and saved investigations.
if curl --fail --silent --max-time 3 http://127.0.0.1:8000/api/repos >/dev/null 2>&1; then
  printf '\naction-trace is already running: %s\n' "$demo_url"
  if "$demo_open"; then open "$demo_url"; fi
  exit 0
fi
if [ -x .venv-product/bin/python ]; then
  product_python=.venv-product/bin/python
elif [ -x .venv-eval/bin/python ]; then
  product_python=.venv-eval/bin/python
else
  echo 'Create .venv-product and install the dependencies documented in docs/PRODUCT.md.' >&2
  exit 1
fi
npm --prefix product_web run build
printf '\nOpen http://127.0.0.1:8000/?mode=compare\n\n'
if "$demo_open"; then
  (
    attempts=0
    until curl --fail --silent --max-time 1 http://127.0.0.1:8000/api/repos >/dev/null 2>&1; do
      attempts=$((attempts + 1))
      [ "$attempts" -lt 30 ] || exit 0
      sleep 1
    done
    open "$demo_url"
  ) &
fi
exec "$product_python" -u -m product_api
