#!/usr/bin/env bash
# Vendor tamper gate (26Q3-REPO vendoring rule 3): every file under
# flaime_serving/vendored/ (except __init__.py) must appear in
# VENDORED_FROM.json with a matching sha256. No network, no secrets.
set -euo pipefail
cd "$(dirname "$0")/.."
fail=0
while IFS=$'\t' read -r path expected; do
  [ -f "$path" ] || { echo "MISSING: $path is listed in the manifest but absent"; fail=1; continue; }
  actual=$(sha256sum "$path" | cut -d' ' -f1)
  [ "$actual" = "$expected" ] || { echo "TAMPERED: $path (sha256 $actual != manifest $expected)"; fail=1; }
done < <(python3 -c 'import json; [print(f["path"], f["sha256"], sep="\t") for f in json.load(open("VENDORED_FROM.json"))["files"]]')
scan_dirs="flaime_serving/vendored"
[ -d tests/vendored ] && scan_dirs="$scan_dirs tests/vendored"
while IFS= read -r f; do
  grep -Fq "\"$f\"" VENDORED_FROM.json || { echo "UNLISTED vendored file: $f"; fail=1; }
done < <(find $scan_dirs -name '*.py' ! -name '__init__.py')
exit "$fail"
