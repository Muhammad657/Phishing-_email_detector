"""Generate a labeled synthetic email dataset (phishing vs. legitimate).

Real phishing corpora (e.g. the Nazario/PhishTank sets, Enron for ham) are the
right thing to train on in production, but they need downloading and licensing.
To keep this project self-contained and runnable offline, we synthesize a
balanced dataset from randomized templates that mimic the structure of real
phishing lures and everyday legitimate mail.

Run:  python data/make_dataset.py --n 4000 --out data/emails.csv
"""

from __future__ import annotations

import argparse
import csv
import random

# --- building blocks --------------------------------------------------------

BRANDS = ["PayPal", "Apple", "Microsoft", "Amazon", "Netflix", "Chase Bank",
          "Wells Fargo", "DHL", "FedEx", "Google", "LinkedIn", "Instagram"]

FIRST_NAMES = ["Alex", "Sam", "Jordan", "Taylor", "Priya", "Wei", "Maria",
               "David", "Fatima", "John", "Aisha", "Carlos", "Sofia", "Liam"]

PHISH_DOMAINS = [
    "secure-{b}-verify.com", "{b}-account-update.net", "{b}login-support.info",
    "account-{b}.security-check.ru", "{b}-billing.top", "verify-{b}.xyz",
    "{b}.com-login-alert.click", "customer-{b}.support-team.tk",
]

LEGIT_DOMAINS = {
    "PayPal": "paypal.com", "Apple": "apple.com", "Microsoft": "microsoft.com",
    "Amazon": "amazon.com", "Netflix": "netflix.com", "Chase Bank": "chase.com",
    "Wells Fargo": "wellsfargo.com", "DHL": "dhl.com", "FedEx": "fedex.com",
    "Google": "google.com", "LinkedIn": "linkedin.com", "Instagram": "instagram.com",
}

PHISH_SUBJECTS = [
    "URGENT: Your {b} account has been suspended",
    "Action Required: Verify your {b} account within 24 hours",
    "Your {b} payment could not be processed",
    "Final Notice: {b} account will be deactivated",
    "Security Alert: Unusual sign-in to your {b} account",
    "You have (1) pending message from {b}",
    "Congratulations! You've won a $1,000 {b} gift card",
]

PHISH_BODIES = [
    ("Dear Customer,\n\nWe detected unusual activity on your {b} account. "
     "Your account has been temporarily LIMITED. You must verify your identity "
     "immediately or your account will be permanently suspended.\n\n"
     "Click here to restore access: {url}\n\n"
     "Please confirm your password and billing information to avoid suspension.\n\n"
     "{b} Security Team"),
    ("Dear User,\n\nYour {b} payment of $349.99 could not be processed. "
     "To avoid service interruption, update your credit card and card number "
     "right away.\n\nVerify now: {url}\n\nFailure to act within 24 hours will "
     "result in account deactivation.\n\nThank you,\n{b} Billing"),
    ("Congratulations!!!\n\nYou have been selected as the WINNER of our monthly "
     "prize draw. Claim your free gift card now by confirming your details "
     "here: {url}\n\nThis offer expires today. Act now!\n\n{b} Rewards"),
    ("Dear Account Holder,\n\nWe were unable to verify your recent login. "
     "For your security, your account has been locked. Confirm your identity "
     "and password here: {url}\n\nIf you do not verify within 24 hours your "
     "account will be closed.\n\n{b} Support"),
]

LEGIT_SUBJECTS = [
    "Your {b} receipt for order #{num}",
    "{b}: Your monthly statement is ready",
    "Welcome to {b}, {name}!",
    "Your {b} password was changed",
    "Meeting notes from today",
    "Re: Lunch on Friday?",
    "Your package has shipped",
    "Project update and next steps",
    "Invoice #{num} from {b}",
]

LEGIT_BODIES = [
    ("Hi {name},\n\nThanks for your recent purchase. Your order #{num} has "
     "shipped and should arrive in 3-5 business days. You can track it anytime "
     "from your account at {url}.\n\nBest regards,\nThe {b} Team"),
    ("Hi {name},\n\nJust following up on our conversation earlier. I've attached "
     "the notes from today's meeting. Let me know if you'd like to move the "
     "review to Thursday.\n\nThanks,\nJordan"),
    ("Hello {name},\n\nYour {b} monthly statement is now available. You can view "
     "it by signing in to {url}. No action is needed -- this is just a heads up.\n\n"
     "Regards,\n{b}"),
    ("Hey {name},\n\nAre we still on for lunch Friday? I was thinking the place "
     "near the office around noon. Let me know what works for you.\n\nCheers,\nSam"),
    ("Hi team,\n\nQuick update: the migration finished over the weekend and "
     "everything looks stable. Next step is to close out the remaining tickets. "
     "I'll send a summary by end of day.\n\nThanks,\nMaria"),
    ("Hi {name},\n\nThis is a confirmation that your password was changed. "
     "If this was you, no further action is required. If you didn't make this "
     "change, please contact support through {url}.\n\n{b} Accounts"),
]


def _phish_url(brand: str) -> str:
    b = brand.lower().replace(" ", "")
    dom = random.choice(PHISH_DOMAINS).format(b=b)
    # Sometimes use a raw IP address, a classic phishing tell.
    if random.random() < 0.25:
        ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
        return f"http://{ip}/{b}/login"
    path = random.choice(["login", "verify", "secure/update", "account/confirm"])
    return f"http://{dom}/{path}"


def _legit_url(brand: str) -> str:
    dom = LEGIT_DOMAINS.get(brand, "example.com")
    path = random.choice(["account", "orders", "help", "settings"])
    return f"https://www.{dom}/{path}"


def make_phishing() -> str:
    b = random.choice(BRANDS)
    subject = random.choice(PHISH_SUBJECTS).format(b=b)
    body = random.choice(PHISH_BODIES).format(b=b, url=_phish_url(b))
    return f"Subject: {subject}\n\n{body}"


def make_legit() -> str:
    b = random.choice(BRANDS)
    ctx = {
        "b": b,
        "name": random.choice(FIRST_NAMES),
        "num": random.randint(10000, 99999),
        "url": _legit_url(b),
    }
    subject = random.choice(LEGIT_SUBJECTS).format(**ctx)
    body = random.choice(LEGIT_BODIES).format(**ctx)
    return f"Subject: {subject}\n\n{body}"


def build(n: int, seed: int = 42) -> list[tuple[str, int]]:
    random.seed(seed)
    rows: list[tuple[str, int]] = []
    for _ in range(n // 2):
        rows.append((make_phishing(), 1))
        rows.append((make_legit(), 0))
    random.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic phishing dataset")
    ap.add_argument("--n", type=int, default=4000, help="total number of emails")
    ap.add_argument("--out", default="data/emails.csv", help="output CSV path")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build(args.n, args.seed)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])  # label: 1 = phishing, 0 = legitimate
        w.writerows(rows)

    n_phish = sum(lbl for _, lbl in rows)
    print(f"Wrote {len(rows)} emails to {args.out} "
          f"({n_phish} phishing / {len(rows) - n_phish} legitimate)")


if __name__ == "__main__":
    main()
