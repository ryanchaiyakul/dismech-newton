# full gates per option, sequential (one GPU job)
for cfg in "0 lagged" "1 lagged" "0 staggered"; do
  set -- $cfg
  echo "== DM_PROXY=$1 DM_MODE=$2"
  DM_PROXY=$1 DM_MODE=$2 uv run pytest tests/test_coupling.py -m "slow or not slow" -q --tb=line -p no:cacheprovider 2>&1 | grep -E "passed|failed|Error|^E |\.py:[0-9]+:" | grep -v Warning
  for i in 1 2 3 4; do
    DM_PROXY=$1 DM_MODE=$2 uv run examples/newton/cloth_rigid_trampoline.py --viewer null --test 2>&1 | grep -iE "assert|passed|fail|error" | tail -2 | sed "s/^/  tramp$i: /"
    echo "  tramp$i exit $?"
  done
done
