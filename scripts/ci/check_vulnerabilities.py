#!/usr/bin/env python3
"""
Check Safety vulnerability scan results and fail on HIGH/CRITICAL.

Usage:
    python scripts/ci/check_vulnerabilities.py safety-report.json
"""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_vulnerabilities.py <safety-report.json>")
        sys.exit(1)

    report_path = Path(sys.argv[1])

    if not report_path.exists():
        print(f"❌ Report file not found: {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        report = json.load(f)

    vulnerabilities = report.get('vulnerabilities', [])

    if not vulnerabilities:
        print("✅ No vulnerabilities found!")
        return 0

    # Filter HIGH and CRITICAL severities
    critical = [
        v for v in vulnerabilities
        if v.get('severity', '').upper() in ['HIGH', 'CRITICAL']
    ]

    if critical:
        print(f"❌ Found {len(critical)} HIGH/CRITICAL vulnerabilities:")
        print()
        for v in critical:
            print(f"  Package: {v.get('package', 'unknown')} {v.get('version', 'unknown')}")
            print(f"  Severity: {v.get('severity', 'unknown')}")
            print(f"  Advisory: {v.get('advisory', 'No details')}")
            print(f"  CVE: {v.get('cve', 'N/A')}")
            print()
        sys.exit(1)
    else:
        low_medium = len(vulnerabilities)
        print(f"✅ No HIGH/CRITICAL vulnerabilities (found {low_medium} LOW/MEDIUM)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
