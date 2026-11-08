# per option: bounce.py and test_pinned.py's verdict (one GPU job at a time)
for cfg in "0 lagged" "1 lagged" "2 lagged" "0 staggered" "1 staggered"; do
  set -- $cfg
  echo "== DM_PROXY=$1 DM_MODE=$2 DM_ITER=${DM_ITER:-1}"
  DM_PROXY=$1 DM_MODE=$2 uv run python exp/step4/bounce.py 2>&1 | grep BOUNCE
  DM_PROXY=$1 DM_MODE=$2 uv run pytest -q -s exp/step4/test_pinned.py -p no:cacheprovider 2>&1 | grep -E "^z |passed|failed|Error" | head -4
done
