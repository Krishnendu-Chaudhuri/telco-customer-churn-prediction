"""CLI for issuing and revoking per-client API keys."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_bootstrap_root = Path(__file__).resolve().parents[1]
if str(_bootstrap_root) not in sys.path:
    sys.path.insert(0, str(_bootstrap_root))

from src.utils.paths import ensure_project_imports

PROJECT_ROOT = ensure_project_imports(Path(__file__))

from src.utils.api_keys import issue_key, list_keys, revoke_by_prefix


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage per-client API keys")
    subparsers = parser.add_subparsers(dest="command", required=True)

    issue_parser = subparsers.add_parser("issue", help="Issue a new API key")
    issue_parser.add_argument("client_name", help="Client identifier")
    issue_parser.add_argument(
        "--scopes",
        default="predict",
        help="Comma-separated scopes: predict (inference) or admin (train/rollback)",
    )

    issue_admin_parser = subparsers.add_parser(
        "issue-admin",
        help="Issue a new API key with admin scope",
    )
    issue_admin_parser.add_argument("client_name", help="Client identifier")

    revoke_parser = subparsers.add_parser("revoke", help="Revoke keys by prefix")
    revoke_parser.add_argument("key_prefix", help="Plaintext key prefix")

    subparsers.add_parser("list", help="List stored API keys")

    args = parser.parse_args()

    if args.command == "issue":
        raw_key = issue_key(args.client_name, scopes=args.scopes)
        print(f"client={args.client_name}")
        print(f"scopes={args.scopes}")
        print(f"key={raw_key}")
        print("Store this key securely; it will not be shown again.")
        return 0

    if args.command == "issue-admin":
        raw_key = issue_key(args.client_name, scopes="admin")
        print(f"client={args.client_name}")
        print("scopes=admin")
        print(f"key={raw_key}")
        print("Store this key securely; it will not be shown again.")
        return 0

    if args.command == "revoke":
        revoked = revoke_by_prefix(args.key_prefix)
        print(f"revoked={revoked}")
        return 0

    if args.command == "list":
        for entry in list_keys():
            status = "active" if entry["revoked_at"] is None else "revoked"
            print(
                f"{entry['key_prefix']}  {entry['client_name']}  "
                f"{entry['scopes']}  {status}  {entry['created_at']}"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
