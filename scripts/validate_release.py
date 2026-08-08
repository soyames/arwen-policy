import argparse
from pathlib import Path

from arwen_etl.release import validate_manifest

parser = argparse.ArgumentParser()
parser.add_argument("release")
args = parser.parse_args()

errors = validate_manifest(
    Path(args.release) / "RELEASE_MANIFEST.json"
)

if errors:
    raise SystemExit("\n".join(errors))

print("Release valid.")