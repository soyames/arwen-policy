import argparse
import json

from arwen_etl.cli import process_discovered_url

parser = argparse.ArgumentParser()
parser.add_argument("url")
args = parser.parse_args()

print(json.dumps(process_discovered_url(args.url), indent=2))