"""Web UI for the phishing email detector.

A dependency-free web app (Python standard library only -- no Flask/Django) that
serves a single page where you paste an email's subject and body and get back a
verdict, a phishing-probability gauge, and the interpretable signals that fired.

Run:
    python app.py                 # then open http://127.0.0.1:8000
    python app.py --port 9000
    python app.py --model models/phishing_model.joblib

The heavy lifting (model + engineered signals) is reused from src/, so the UI is
just a thin presentation layer over the same `classify()` the CLI uses.
"""
from __future__ import annotations


import argparse
import json
import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.predict import DEFAULT_MODEL, classify, load_model
from src.gmail_imap import GmailError, fetch_recent
from src.explain_ai import explain_phishing_result   # ADD THIS

# Loaded once at startup and shared across request threads (read-only use).
MODEL = None
MODEL_PATH = DEFAULT_MODEL

# Gmail credentials, read from the environment / .env at startup (never stored
# on disk by us, never logged). Empty until configured by the user.
GMAIL_ADDRESS = ""
GMAIL_APP_PASSWORD = ""


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines) -- avoids a python-dotenv dep.

    Values already present in the real environment win over the file.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

# Pre-filled examples the "Try an example" buttons drop into the form.
EXAMPLES = [
    {
        "label": "PayPal scam",
        "subject": "URGENT: Your PayPal account has been suspended",
        "body": (
            "Dear Customer,\n\nWe detected unusual activity on your account, "
            "which has been LIMITED. You must verify your password and card "
            "number immediately or your account will be permanently suspended.\n\n"
            "Confirm your identity here: http://secure-paypal-verify.top/login\n\n"
            "PayPal Security Team"
        ),
    },
    {
        "label": "Lottery lure",
        "subject": "Congratulations!!! You have WON $1,000,000",
        "body": (
            "Dear Winner,\n\nYour email address was selected in our international "
            "lottery draw. To claim your prize of $1,000,000 you must act now and "
            "send your full name, bank account number and a processing fee via "
            "wire transfer.\n\nClaim at http://192.168.44.9/claim-reward"
        ),
    },
    {
        "label": "Normal work email",
        "subject": "Lunch on Friday?",
        "body": (
            "Hey Sam,\n\nAre we still on for lunch this Friday around noon at the "
            "usual place? Let me know what works for you.\n\nCheers,\nAlex"
        ),
    },
    {
        "label": "Real receipt",
        "subject": "Your Amazon receipt for order #48213",
        "body": (
            "Hi Taylor,\n\nThanks for your purchase. Your order has shipped and "
            "will arrive in 3-5 business days. You can track it any time from "
            "your orders page at https://www.amazon.com/orders.\n\nThe Amazon Team"
        ),
    },
]

# Human-friendly names + descriptions for the engineered signals in features.py.
SIGNAL_INFO = {
    "n_urls": ("Links in the email", "The message contains one or more URLs."),
    "n_ip_urls": ("Raw IP-address links", "Links point at a bare IP address instead of a domain name -- a classic phishing tell."),
    "n_emails": ("Embedded email addresses", "Email addresses appear inside the body."),
    "urgency": ("Urgency / pressure language", "Words like 'urgent', 'immediately', 'suspended', 'act now' that rush you into acting."),
    "credential_request": ("Requests credentials", "Asks for passwords, card numbers, PINs or other sensitive details."),
    "lure": ("Too-good-to-be-true lure", "Prizes, refunds, lottery wins or other bait."),
    "generic_greeting": ("Generic greeting", "'Dear Customer' style greeting instead of your name."),
    "brand_mismatch": ("Brand / link mismatch", "Names a known brand but links to an unrelated domain -- likely spoofing."),
    "suspicious_tld": ("Suspicious domain ending", "Links use a domain ending frequently abused by phishers (.top, .xyz, .tk, ...)."),
    "lookalike_domain": ("Look-alike domain", "A domain with digits swapped for letters, e.g. paypa1 or g00gle."),
    "n_exclaim": ("Lots of exclamation marks", "Excessive '!' -- common in scam mail."),
}


def humanize_signals(signals: list) -> list[dict]:
    """Turn (name, value) signal tuples into rich objects for the frontend."""
    out = []
    for name, value in signals:
        title, desc = SIGNAL_INFO.get(name, (name.replace("_", " ").title(), ""))
        out.append({"key": name, "title": title, "desc": desc, "value": value})
    return out


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phishing Email Detector</title>
<style>
  :root{
    --bg:#0f172a; --panel:#1e293b; --panel-2:#243349; --line:#334155;
    --text:#e2e8f0; --muted:#94a3b8; --accent:#38bdf8;
    --safe:#22c55e; --warn:#f59e0b; --danger:#ef4444;
    --radius:14px; --shadow:0 10px 30px rgba(0,0,0,.35);
  }
  *{box-sizing:border-box}
  body{
    margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    background:radial-gradient(1200px 600px at 80% -10%,#1d3a5f 0%,transparent 60%),var(--bg);
    color:var(--text); min-height:100vh; line-height:1.5;
  }
  .wrap{max-width:920px; margin:0 auto; padding:32px 20px 64px}
  header{display:flex; align-items:center; gap:14px; margin-bottom:8px}
  .logo{font-size:34px; line-height:1}
  h1{font-size:24px; margin:0; letter-spacing:.2px}
  .sub{color:var(--muted); margin:2px 0 24px; font-size:14px}
  .grid{display:grid; grid-template-columns:1fr; gap:20px}
  @media(min-width:820px){.grid{grid-template-columns:1fr 1fr}}
  .card{
    background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
    padding:20px; box-shadow:var(--shadow);
  }
  label{display:block; font-size:13px; color:var(--muted); margin:0 0 6px; font-weight:600}
  input[type=text], textarea{
    width:100%; background:var(--panel-2); border:1px solid var(--line);
    color:var(--text); border-radius:10px; padding:11px 12px; font-size:14px;
    font-family:inherit; resize:vertical;
  }
  input:focus, textarea:focus{outline:none; border-color:var(--accent); box-shadow:0 0 0 3px rgba(56,189,248,.15)}
  textarea{min-height:220px}
  .field{margin-bottom:16px}
  .row{display:flex; gap:10px; flex-wrap:wrap; align-items:center}
  button{
    cursor:pointer; border:none; border-radius:10px; font-size:14px; font-weight:600;
    padding:11px 18px; transition:.15s transform,.15s background;
  }
  button:active{transform:translateY(1px)}
  .btn-primary{background:var(--accent); color:#03263a}
  .btn-primary:hover{background:#5cccfa}
  .btn-ghost{background:transparent; color:var(--muted); border:1px solid var(--line); padding:7px 12px; font-size:12px}
  .btn-ghost:hover{color:var(--text); border-color:var(--accent)}
  .examples{margin:14px 0 0}
  .examples .lbl{font-size:12px; color:var(--muted); margin-bottom:8px}
  .placeholder{color:var(--muted); text-align:center; padding:40px 10px}
  .placeholder .big{font-size:40px; opacity:.5}
  /* result */
  .verdict{display:flex; align-items:center; gap:14px; margin-bottom:18px}
  .badge{
    font-weight:800; font-size:15px; letter-spacing:.5px; padding:8px 14px;
    border-radius:999px; white-space:nowrap;
  }
  .verdict h2{margin:0; font-size:20px}
  .verdict .risk{color:var(--muted); font-size:13px; margin-top:2px}
  .gauge-label{display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-bottom:6px}
  .gauge{height:14px; border-radius:999px; background:var(--panel-2); overflow:hidden; border:1px solid var(--line)}
  .gauge > span{display:block; height:100%; width:0; transition:width .6s ease}
  .pct{font-size:34px; font-weight:800; margin:14px 0 2px}
  .pct small{font-size:14px; color:var(--muted); font-weight:600}
  .signals{margin-top:20px}
  .signals h3{font-size:13px; text-transform:uppercase; letter-spacing:.6px; color:var(--muted); margin:0 0 10px}
  .sig{
    display:flex; gap:10px; align-items:flex-start; padding:10px 12px; margin-bottom:8px;
    background:var(--panel-2); border:1px solid var(--line); border-left:3px solid var(--warn);
    border-radius:8px;
  }
  .sig .dot{color:var(--warn); font-size:16px; line-height:1.2}
  .sig .t{font-weight:700; font-size:13px}
  .sig .d{font-size:12px; color:var(--muted)}
  .sig .v{margin-left:auto; font-size:12px; color:var(--muted); white-space:nowrap}
  .none{color:var(--muted); font-size:13px; font-style:italic}
  .spinner{width:26px;height:26px;border:3px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin .8s linear infinite;margin:30px auto}
  @keyframes spin{to{transform:rotate(360deg)}}
  footer{margin-top:28px; color:var(--muted); font-size:12px; text-align:center}
  .err{color:var(--danger); font-size:13px}
  /* tabs */
  .tabs{display:flex; gap:6px; margin-bottom:20px; border-bottom:1px solid var(--line)}
  .tab{background:transparent; color:var(--muted); border:none; border-bottom:2px solid transparent; border-radius:0; padding:10px 16px; font-size:14px; font-weight:600; cursor:pointer}
  .tab:hover{color:var(--text)}
  .tab.active{color:var(--accent); border-bottom-color:var(--accent)}
  .pane{display:none}
  .pane.active{display:block}
  /* inbox */
  .connect{background:var(--panel-2); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:16px}
  .connect .field{margin-bottom:12px}
  .connect .field:last-of-type{margin-bottom:0}
  .connect a{color:var(--accent)}
  .controls{display:flex; gap:14px; align-items:flex-end; flex-wrap:wrap; margin-bottom:8px}
  .controls .field{margin:0}
  .controls input[type=number]{width:90px}
  .check{display:flex; align-items:center; gap:8px; color:var(--text); font-size:14px; user-select:none}
  .check input{width:16px; height:16px; accent-color:var(--accent)}
  .hint{font-size:12px; color:var(--muted); margin-top:12px}
  .hint code{background:var(--panel-2); padding:1px 6px; border-radius:5px; border:1px solid var(--line)}
  .status{font-size:13px; color:var(--muted); margin:14px 0 6px}
  .summary{display:flex; gap:16px; flex-wrap:wrap; margin:10px 0 16px}
  .stat{background:var(--panel-2); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:90px}
  .stat .num{font-size:22px; font-weight:800}
  .stat .cap{font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.5px}
  .mail{border:1px solid var(--line); border-left-width:4px; border-radius:10px; background:var(--panel-2); margin-bottom:10px; overflow:hidden}
  .mail-head{display:flex; gap:12px; align-items:center; padding:12px 14px; cursor:pointer}
  .mail-head:hover{background:#2a3b54}
  .mail .pill{font-size:11px; font-weight:800; padding:4px 9px; border-radius:999px; white-space:nowrap}
  .mail .meta{min-width:0; flex:1}
  .mail .subj{font-weight:700; font-size:14px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .mail .from{font-size:12px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
  .mail .prob{font-size:13px; font-weight:700; white-space:nowrap}
  .mail .chev{color:var(--muted); font-size:12px; transition:transform .2s}
  .mail.open .chev{transform:rotate(90deg)}
  .mail-body{display:none; padding:0 14px 14px; border-top:1px solid var(--line)}
  .mail.open .mail-body{display:block}
  .mail-body .bodytext{white-space:pre-wrap; font-size:12px; color:var(--muted); background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; max-height:220px; overflow:auto; margin:12px 0}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="logo">🛡️</div>
    <div>
      <h1>Phishing Email Detector</h1>
      <div class="sub">Paste an email and the model flags phishing, with the signals behind its verdict.</div>
    </div>
  </header>

  <div class="tabs">
    <button class="tab active" data-tab="paste">✍️ Paste an email</button>
    <button class="tab" data-tab="inbox">📥 Scan my inbox</button>
  </div>

  <!-- ============ PASTE PANE ============ -->
  <div class="pane active" id="pane-paste">
  <div class="grid">
    <!-- INPUT -->
    <div class="card">
      <form id="form">
        <div class="field">
          <label for="subject">Subject</label>
          <input type="text" id="subject" name="subject" placeholder="e.g. Your account has been suspended">
        </div>
        <div class="field">
          <label for="body">Body</label>
          <textarea id="body" name="body" placeholder="Paste the full email body here..."></textarea>
        </div>
        <div class="row">
          <button type="submit" class="btn-primary">Analyze email</button>
          <button type="button" class="btn-ghost" id="clear">Clear</button>
        </div>
        <div class="examples">
          <div class="lbl">Try an example:</div>
          <div class="row" id="examples"></div>
        </div>
      </form>
    </div>

    <!-- OUTPUT -->
    <div class="card" id="result">
      <div class="placeholder">
        <div class="big">📨</div>
        <p>Your analysis will appear here.</p>
      </div>
    </div>
  </div>
  </div><!-- /pane-paste -->

  <!-- ============ INBOX PANE ============ -->
  <div class="pane" id="pane-inbox">
    <div class="card">
      <div class="connect" id="connect">
        <div class="field">
          <label for="gmail">Gmail address</label>
          <input type="text" id="gmail" placeholder="you@gmail.com" autocomplete="username">
        </div>
        <div class="field">
          <label for="apppw">Google App Password (16 characters)</label>
          <input type="password" id="apppw" placeholder="xxxx xxxx xxxx xxxx" autocomplete="off">
        </div>
        <div class="hint" id="cred-hint"></div>
      </div>

      <div class="controls">
        <div class="field">
          <label for="limit">How many recent emails</label>
          <input type="number" id="limit" min="1" max="100" value="15">
        </div>
        <label class="check"><input type="checkbox" id="unread"> Unread only</label>
        <button type="button" class="btn-primary" id="scan">📥 Scan inbox</button>
      </div>
      <div id="inbox-status"></div>
      <div id="inbox-results"></div>
    </div>
  </div>

  <footer>
    Educational demo · Logistic-regression model over TF-IDF + engineered signals.
    Verdicts are probabilistic — always use human judgment.
  </footer>
</div>

<script>
const EXAMPLES = __EXAMPLES__;
const exWrap = document.getElementById('examples');
EXAMPLES.forEach((ex, i) => {
  const b = document.createElement('button');
  b.type = 'button'; b.className = 'btn-ghost'; b.textContent = ex.label;
  b.onclick = () => {
    document.getElementById('subject').value = ex.subject;
    document.getElementById('body').value = ex.body;
  };
  exWrap.appendChild(b);
});

document.getElementById('clear').onclick = () => {
  document.getElementById('subject').value = '';
  document.getElementById('body').value = '';
};

const colorFor = (verdict) => {
  if (verdict === 'PHISHING') return 'var(--danger)';
  if (verdict === 'SUSPICIOUS') return 'var(--warn)';
  return 'var(--safe)';
};
const iconFor = (verdict) => {
  if (verdict === 'PHISHING') return '⚠️';
  if (verdict === 'SUSPICIOUS') return '🤔';
  return '✅';
};

const resultEl = document.getElementById('result');

document.getElementById('form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const subject = document.getElementById('subject').value;
  const body = document.getElementById('body').value;
  if (!subject.trim() && !body.trim()) {
    resultEl.innerHTML = '<div class="placeholder"><div class="big">📭</div><p>Enter a subject or body first.</p></div>';
    return;
  }
  resultEl.innerHTML = '<div class="spinner"></div>';
  try {
    const r = await fetch('/api/classify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({subject, body})
    });
    if (!r.ok) throw new Error('Server error ' + r.status);
    render(await r.json());
  } catch (err) {
    resultEl.innerHTML = '<p class="err">Could not analyze: ' + err.message + '</p>';
  }
});

function render(res) {
  const color = colorFor(res.verdict);
  const pct = (res.phishing_probability * 100);
  const pctStr = pct.toFixed(1) + '%';

  let sigHtml = '';
  if (res.signals && res.signals.length) {
    sigHtml = res.signals.map(s => `
      <div class="sig" style="border-left-color:${color}">
        <div class="dot" style="color:${color}">●</div>
        <div>
          <div class="t">${s.title}</div>
          <div class="d">${s.desc}</div>
        </div>
        <div class="v">×${(+s.value).toLocaleString(undefined,{maximumFractionDigits:0})}</div>
      </div>`).join('');
  } else {
    sigHtml = '<div class="none">No individual red-flag signals fired.</div>';
  }

  let aiHtml = '';
  if (res.ai_explanation && !res.ai_explanation.error) {
    const ai = res.ai_explanation;
    const flagsHtml = (ai.red_flags && ai.red_flags.length)
      ? ai.red_flags.map(f => `<div class="sig" style="border-left-color:${color}">
          <div class="dot" style="color:${color}">●</div>
          <div><div class="t">${escapeHtml(f)}</div></div>
        </div>`).join('')
      : '';
    aiHtml = `
      <div class="signals" style="margin-top:20px">
        <h3>AI explanation</h3>
        <p>${escapeHtml(ai.summary || '')}</p>
        ${ai.link_warning ? `<p style="color:${color}">⚠ ${escapeHtml(ai.link_warning)}</p>` : ''}
        ${flagsHtml}
        <p style="margin-top:10px">${escapeHtml(ai.explanation || '')}</p>
      </div>`;
  }

  resultEl.innerHTML = `
      <div class="verdict">
        <span class="badge" style="background:${color}22; color:${color}; border:1px solid ${color}">
          ${iconFor(res.verdict)} ${res.verdict}
        </span>
        <div>
          <h2 style="color:${color}">${res.verdict === 'LEGITIMATE' ? 'Looks legitimate' : (res.verdict === 'SUSPICIOUS' ? 'Be cautious' : 'Likely phishing')}</h2>
          <div class="risk">Risk level: ${res.risk}</div>
        </div>
      </div>

      <div class="gauge-label"><span>Phishing probability</span><span>${pctStr}</span></div>
      <div class="gauge"><span style="background:${color}"></span></div>
      <div class="pct" style="color:${color}">${pctStr} <small>chance this is phishing</small></div>

      <div class="signals">
        <h3>Why — signals detected</h3>
        ${sigHtml}
      </div>
      ${aiHtml}`;

    // animate the gauge after it's in the DOM
    requestAnimationFrame(() => {
      resultEl.querySelector('.gauge > span').style.width = Math.min(100, pct) + '%';
    });
  }

/* ---------------- tabs ---------------- */
document.querySelectorAll('.tab').forEach(t => {
  t.onclick = () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.pane').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    document.getElementById('pane-' + t.dataset.tab).classList.add('active');
  };
});

/* ---------------- inbox scan ---------------- */
const GMAIL_CONFIGURED = __GMAIL_CONFIGURED__;
const GMAIL_ADDRESS = "__GMAIL_ADDRESS__";
const credHint = document.getElementById('cred-hint');
const gmailInput = document.getElementById('gmail');
const apppwInput = document.getElementById('apppw');
if (GMAIL_ADDRESS) gmailInput.value = GMAIL_ADDRESS;
credHint.innerHTML =
  'Read-only — nothing in your mailbox is modified. Needs a Google <b>App Password</b> ' +
  '(not your normal password): turn on 2-Step Verification, then create one at ' +
  '<a href="https://myaccount.google.com/apppasswords" target="_blank" rel="noopener">myaccount.google.com/apppasswords</a>. ' +
  'Credentials stay on your machine and are only used for this scan.' +
  (GMAIL_CONFIGURED ? ' <br>A password is already loaded from your <code>.env</code> file — leave the password box blank to use it.' : '');

const escapeHtml = (s) => (s || '').replace(/[&<>"']/g, c => (
  {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const statusEl = document.getElementById('inbox-status');
const inboxResults = document.getElementById('inbox-results');

document.getElementById('scan').onclick = async () => {
  const limit = parseInt(document.getElementById('limit').value || '15', 10);
  const onlyUnread = document.getElementById('unread').checked;
  const address = gmailInput.value.trim();
  // Gmail shows app passwords with spaces; they must be stripped before login.
  const appPassword = apppwInput.value.replace(/\s+/g, '');
  if (!address && !GMAIL_CONFIGURED) {
    statusEl.innerHTML = '<p class="err">Enter your Gmail address first.</p>';
    return;
  }
  if (!appPassword && !GMAIL_CONFIGURED) {
    statusEl.innerHTML = '<p class="err">Enter your Google App Password first.</p>';
    return;
  }
  statusEl.innerHTML = '<div class="spinner"></div>';
  inboxResults.innerHTML = '';
  try {
    const r = await fetch('/api/scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({limit, onlyUnread, address, appPassword})
    });
    const data = await r.json();
    if (!r.ok || data.error) throw new Error(data.error || ('Server error ' + r.status));
    renderInbox(data.emails);
  } catch (err) {
    statusEl.innerHTML = '<p class="err">' + escapeHtml(err.message) + '</p>';
  }
};

function renderInbox(emails) {
  if (!emails.length) {
    statusEl.innerHTML = '<p class="status">No emails found for that filter.</p>';
    return;
  }
  const phish = emails.filter(e => e.verdict === 'PHISHING').length;
  const susp  = emails.filter(e => e.verdict === 'SUSPICIOUS').length;
  const legit = emails.filter(e => e.verdict === 'LEGITIMATE').length;

  statusEl.innerHTML = `
    <div class="summary">
      <div class="stat"><div class="num">${emails.length}</div><div class="cap">Scanned</div></div>
      <div class="stat"><div class="num" style="color:var(--danger)">${phish}</div><div class="cap">Phishing</div></div>
      <div class="stat"><div class="num" style="color:var(--warn)">${susp}</div><div class="cap">Suspicious</div></div>
      <div class="stat"><div class="num" style="color:var(--safe)">${legit}</div><div class="cap">Legitimate</div></div>
    </div>`;

  // most dangerous first
  const order = {PHISHING:0, SUSPICIOUS:1, LEGITIMATE:2};
  emails.sort((a,b) => (order[a.verdict]-order[b.verdict]) || (b.phishing_probability-a.phishing_probability));

  inboxResults.innerHTML = emails.map((e,i) => {
    const color = colorFor(e.verdict);
    const pct = (e.phishing_probability*100).toFixed(0) + '%';
    const sigs = (e.signals && e.signals.length)
      ? e.signals.map(s => `<div class="sig" style="border-left-color:${color}">
            <div class="dot" style="color:${color}">●</div>
            <div><div class="t">${escapeHtml(s.title)}</div><div class="d">${escapeHtml(s.desc)}</div></div>
            <div class="v">×${Math.round(+s.value)}</div></div>`).join('')
      : '<div class="none">No individual red-flag signals fired.</div>';
    return `
      <div class="mail" style="border-left-color:${color}" data-i="${i}">
        <div class="mail-head">
          <span class="pill" style="background:${color}22;color:${color};border:1px solid ${color}">${iconFor(e.verdict)} ${e.verdict}</span>
          <div class="meta">
            <div class="subj">${escapeHtml(e.subject)}</div>
            <div class="from">${escapeHtml(e.sender)}${e.date ? ' · ' + escapeHtml(e.date) : ''}</div>
          </div>
          <span class="prob" style="color:${color}">${pct}</span>
          <span class="chev">▶</span>
        </div>
        <div class="mail-body">
          <div class="signals"><h3>Why — signals detected</h3>${sigs}</div>
          <div class="bodytext">${escapeHtml(e.body_preview || '(no readable text)')}</div>
        </div>
      </div>`;
  }).join('');

  inboxResults.querySelectorAll('.mail-head').forEach(h => {
    h.onclick = () => h.parentElement.classList.toggle('open');
  });
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - stdlib API
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            html = (
                PAGE.replace("__EXAMPLES__", json.dumps(EXAMPLES))
                .replace("__GMAIL_CONFIGURED__", "true" if GMAIL_APP_PASSWORD else "false")
                .replace("__GMAIL_ADDRESS__", GMAIL_ADDRESS or "")
            )
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_POST(self):  # noqa: N802 - stdlib API
        path = urlparse(self.path).path
        if path == "/api/classify":
            self._handle_classify()
        elif path == "/api/scan":
            self._handle_scan()
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def _handle_classify(self):
        try:
            data = self._read_json()
            subject = str(data.get("subject", ""))
            body = str(data.get("body", ""))
            text = (subject + "\n" + body).strip()
            result = classify(MODEL, text)
            result["signals"] = humanize_signals(result["signals"])
            is_phishing = result["verdict"] in ("PHISHING", "SUSPICIOUS")
            ai_result = explain_phishing_result(
                text, is_phishing, confidence=result["phishing_probability"]
            )
            result["ai_explanation"] = ai_result
        
            self._send(200, json.dumps(result).encode("utf-8"), "application/json")
        except Exception as exc:  # keep the server alive, report to client
            err = json.dumps({"error": str(exc)}).encode("utf-8")
            self._send(500, err, "application/json")

    def _handle_scan(self):
        try:
            data = self._read_json()
            limit = int(data.get("limit", 15))
            only_unread = bool(data.get("onlyUnread", False))
            # Prefer credentials typed into the form; fall back to .env values.
            address = str(data.get("address", "")).strip() or GMAIL_ADDRESS
            app_password = str(data.get("appPassword", "")).strip() or GMAIL_APP_PASSWORD
            emails = fetch_recent(
                address, app_password,
                limit=limit, only_unread=only_unread,
            )
            out = []
            for em in emails:
                res = classify(MODEL, em.text)
                preview = em.body if len(em.body) <= 600 else em.body[:600] + " ..."
                out.append({
                    "sender": em.sender,
                    "subject": em.subject,
                    "date": em.date,
                    "verdict": res["verdict"],
                    "risk": res["risk"],
                    "phishing_probability": res["phishing_probability"],
                    "signals": humanize_signals(res["signals"]),
                    "body_preview": preview,
                })
            self._send(200, json.dumps({"emails": out}).encode("utf-8"), "application/json")
        except GmailError as exc:  # friendly, expected errors
            self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")
        except Exception as exc:
            self._send(500, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

    def log_message(self, fmt, *args):  # quieter console
        return


def main() -> None:
    global MODEL, MODEL_PATH, GMAIL_ADDRESS, GMAIL_APP_PASSWORD
    ap = argparse.ArgumentParser(description="Web UI for the phishing detector")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    args = ap.parse_args()

    load_dotenv()
    GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS", "").strip()
    GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "").strip()
    if GMAIL_APP_PASSWORD:
        print(f"Gmail inbox scanning enabled for {GMAIL_ADDRESS} (read-only).")
    else:
        print("Gmail not configured -- inbox scanning disabled "
              "(copy .env.example to .env to enable). Paste mode still works.")

    MODEL_PATH = args.model
    print(f"Loading model from {MODEL_PATH} ...")
    MODEL = load_model(MODEL_PATH)  # raises a helpful error if missing

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Phishing Email Detector UI running at  {url}")
    print("  Press Ctrl+C to stop.\n")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()