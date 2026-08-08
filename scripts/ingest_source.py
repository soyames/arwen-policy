import argparse
import json

from arwen_etl.cli import ingest_url

parser = argparse.ArgumentParser()
parser.add_argument("url")
args = parser.parse_args()

print(json.dumps(ingest_url(args.url), indent=2))