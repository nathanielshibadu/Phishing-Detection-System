# server_side_api/test_cli.py
import argparse
from app import is_phishing  # import the helper from app.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to test")
    args = parser.parse_args()
    result = is_phishing(args.url)
    print(result)
