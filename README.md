# 🛡️ Phishing Email Detector

A machine-learning phishing detector with a clean web UI. Paste an email (or scan
your Gmail inbox read-only) and get a verdict, a phishing-probability score, and
the **interpretable signals** behind the decision — urgency language, credential
requests, look-alike domains, brand/link mismatches, and more.

The model is a logistic-regression classifier over **TF-IDF features + hand-engineered
red-flag signals**, which keeps it fast, well-calibrated, and explainable.

---

## Features

- **✍️ Paste mode** — drop in any subject + body and get an instant verdict.
- **📥 Inbox scan** — connect to Gmail over IMAP (read-only) and score your latest
  emails into a ranked, most-dangerous-first list.
- **Explainable** — every verdict lists the concrete signals that fired.
- **No heavy web framework** — the UI is served by the Python standard library
  (`http.server`); the only third-party deps are the ML stack.

---

## Quick start

```bash
# 1. create a virtual environment and install deps
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. build a dataset and train the model
#    (a) self-contained synthetic dataset:
python data/make_dataset.py --n 4000 --out data/emails.csv
#    (b) or bring a real corpus (e.g. CEAS-2008) with sender/subject/body/label columns
python -m src.train --data data/emails.csv --out models/phishing_model.joblib

# 3. run the web UI
python app.py
```

Then open **http://127.0.0.1:8000**.

> The trained model (`models/*.joblib`) and datasets (`data/emails.csv`) are
> git-ignored, so you train them locally with the commands above.

---

## Scanning your Gmail inbox

Inbox scanning is **read-only** — it never sends, deletes, or marks anything in your
mailbox. It uses IMAP, which Gmail requires an **App Password** for (your normal
password won't work).

1. Turn on **2-Step Verification**: https://myaccount.google.com/security
2. Create a 16-character **App Password**: https://myaccount.google.com/apppasswords
   (choose "Mail").
3. In the app, open the **📥 Scan my inbox** tab and enter your Gmail address +
   the app password, then click **Scan inbox**.

Prefer not to type it each time? Copy `.env.example` to `.env` and fill in
`GMAIL_ADDRESS` / `GMAIL_APP_PASSWORD` — the app loads them at startup. **`.env` is
git-ignored so your credentials never leave your machine.**

---

## Command-line usage

The classifier also works from the terminal, without the UI:

```bash
python -m src.predict --text "URGENT: verify your account at http://bit.ly/x"
python -m src.predict --file suspicious.txt
python -m src.predict --demo
```

---

## Project structure

```
app.py                  # web UI (stdlib http.server) — paste mode + inbox scan
src/
  features.py           # TF-IDF + engineered red-flag features
  train.py              # train / evaluate / save the model
  predict.py            # scoring + CLI
  gmail_imap.py         # read-only Gmail IMAP reader
data/make_dataset.py    # synthetic labeled dataset generator
requirements.txt
.env.example            # template for Gmail credentials (copy to .env)
```

---

## Security & privacy

- Gmail access is **read-only**; the mailbox is opened with `readonly=True`.
- Credentials are only held in memory for the duration of a scan and are **never
  logged or committed** (`.env` is git-ignored).
- Scored emails never leave your machine — all inference runs locally.

## Disclaimer

This is an educational project. Verdicts are probabilistic and can be wrong in both
directions — always apply human judgment before acting on any email.
# ML-Phishing-Email-Detector
