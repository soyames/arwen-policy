import argparse

from arwen_etl.release import build_release_manifest

parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True)
args = parser.parse_args()

print(
    build_release_manifest(
        f"data/releases/{args.version}",
        args.version,
    )
)