"""Score emails with the trained phishing detector.

Examples:
    # score a raw email piped in
    cat suspicious.txt | python -m src.predict

    # score a file
    python -m src.predict --file suspicious.txt

    # score inline text
    python -m src.predict --text "URGENT: verify your account at http://bit.ly/x"

    # run the built-in demo emails
    python -m src.predict --demo
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import explain  # noqa: E402

DEFAULT_MODEL = "models/phishing_model.joblib"


def load_model(path: str):
    if not os.path.exists(path):
        raise SystemExit(
            f"Model not found: {path}\n"
            "Train it first:\n"
            "  python data/make_dataset.py --out data/emails.csv\n"
            "  python -m src.train --data data/emails.csv --out " + path
        )
    return joblib.load(path)


def classify(model, text: str) -> dict:
    prob = float(model.predict_proba([text])[0][1])
    if prob >= 0.80:
        verdict, risk = "PHISHING", "high"
    elif prob >= 0.50:
        verdict, risk = "PHISHING", "medium"
    elif prob >= 0.30:
        verdict, risk = "SUSPICIOUS", "low"
    else:
        verdict, risk = "LEGITIMATE", "low"
    return {
        "verdict": verdict,
        "risk": risk,
        "phishing_probability": round(prob, 4),
        "signals": [(e.signal, e.value) for e in explain(text)],
    }


def _print_report(text: str, result: dict) -> None:
    preview = text.strip().replace("\n", " ")
    if len(preview) > 90:
        preview = preview[:90] + "..."
    bar_len = int(result["phishing_probability"] * 30)
    bar = "#" * bar_len + "-" * (30 - bar_len)

    print("-" * 64)
    print(f"Email:   {preview}")
    print(f"Verdict: {result['verdict']}  (risk: {result['risk']})")
    print(f"P(phish):[{bar}] {result['phishing_probability']:.1%}")
    if result["signals"]:
        pretty = ", ".join(f"{name}={val:g}" for name, val in result["signals"])
        print(f"Signals: {pretty}")
    print("-" * 64)


DEMO_EMAILS = [
    ("Subject: URGENT: Your PayPal account has been suspended\n\n"
     "Dear Customer, We detected unusual activity. Your account has been "
     "LIMITED. Verify your password and card number immediately or it will be "
     "permanently suspended: http://secure-paypal-verify.top/login"),
    ("Subject: Lunch on Friday?\n\nHey Sam, are we still on for lunch Friday "
     "around noon at the usual place? Let me know what works. Cheers, Alex"),
    ("Subject: Your Amazon receipt for order #48213\n\nHi Taylor, thanks for "
     "your purchase. Your order has shipped and will arrive in 3-5 days. Track "
     "it at https://www.amazon.com/orders. Best, The Amazon Team"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Classify email(s) as phishing or legitimate")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", help="email text to classify")
    src.add_argument("--file", help="path to a file containing the email")
    src.add_argument("--demo", action="store_true", help="run built-in demo emails")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", action="store_true", help="output raw JSON")
    args = ap.parse_args()

    model = load_model(args.model)

    if args.demo:
        emails = DEMO_EMAILS
    elif args.text is not None:
        emails = [args.text]
    elif args.file is not None:
        with open(args.file, encoding="utf-8", errors="ignore") as f:
            emails = [f.read()]
    else:  # read from stdin
        data = sys.stdin.read()
        if not data.strip():
            ap.error("no input: pass --text, --file, --demo, or pipe text via stdin")
        emails = [data]

    results = [classify(model, e) for e in emails]

    if args.json:
        import json
        payload = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    else:
        for text, res in zip(emails, results):
            _print_report(text, res)


if __name__ == "__main__":
    main()
