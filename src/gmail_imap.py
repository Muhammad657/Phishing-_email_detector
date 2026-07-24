"""Read-only Gmail access over IMAP for the phishing detector UI.

Uses only the Python standard library (`imaplib` + `email`). Nothing here ever
sends, deletes, flags, or otherwise mutates your mailbox -- the mailbox is
opened read-only and we only FETCH message contents.

Credentials are read from the environment (never hard-coded):
    GMAIL_ADDRESS       your full gmail address, e.g. you@gmail.com
    GMAIL_APP_PASSWORD  a 16-char Google *App Password* (not your login password)

App Passwords require 2-Step Verification to be on:
    https://myaccount.google.com/apppasswords
"""
from __future__ import annotations

import email
import html
import imaplib
import re
from dataclasses import dataclass, field
from email.header import decode_header, make_header
from email.message import Message

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993

# Don't feed unbounded text to the model -- cap very long bodies.
MAX_BODY_CHARS = 20000


class GmailError(RuntimeError):
    """Raised for connection / auth / fetch problems, with a friendly message."""


@dataclass
class InboxEmail:
    uid: str
    sender: str
    subject: str
    date: str
    body: str = field(repr=False)

    @property
    def text(self) -> str:
        """Combined subject+body string handed to the classifier."""
        return f"{self.subject}\n{self.body}".strip()


def _decode(value: str | None) -> str:
    """Decode a possibly RFC 2047-encoded header (=?utf-8?...) to plain text."""
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"[ \t]*\n[ \t]*")


def _html_to_text(raw_html: str) -> str:
    no_scripts = _SCRIPT_STYLE_RE.sub(" ", raw_html)
    text = _TAG_RE.sub(" ", no_scripts)
    text = html.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return _WS_RE.sub("\n", text).strip()


def _payload_text(part: Message) -> str:
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    try:
        return payload.decode(charset, errors="replace")
    except (LookupError, TypeError):
        return payload.decode("utf-8", errors="replace")


def _extract_body(msg: Message) -> str:
    """Prefer text/plain; fall back to stripped text/html. Skip attachments."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            ctype = part.get_content_type()
            if ctype == "text/plain":
                plain_parts.append(_payload_text(part))
            elif ctype == "text/html":
                html_parts.append(_payload_text(part))
    else:
        ctype = msg.get_content_type()
        if ctype == "text/html":
            html_parts.append(_payload_text(msg))
        else:
            plain_parts.append(_payload_text(msg))

    if any(p.strip() for p in plain_parts):
        body = "\n".join(plain_parts)
    else:
        body = _html_to_text("\n".join(html_parts))

    body = body.strip()
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + " ..."
    return body


def fetch_recent(
    address: str,
    app_password: str,
    limit: int = 20,
    only_unread: bool = False,
    folder: str = "INBOX",
) -> list[InboxEmail]:
    """Return the most recent `limit` emails from `folder`, newest first.

    The mailbox is opened read-only, so fetching does NOT mark anything as read.
    """
    if not address or not app_password:
        raise GmailError(
            "Gmail credentials are not set. Create a .env file with "
            "GMAIL_ADDRESS and GMAIL_APP_PASSWORD (see .env.example)."
        )

    try:
        conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    except Exception as exc:  # network / DNS
        raise GmailError(f"Could not reach {IMAP_HOST}: {exc}") from exc

    try:
        try:
            conn.login(address, app_password)
        except imaplib.IMAP4.error as exc:
            raise GmailError(
                "Login failed. Check that GMAIL_ADDRESS is correct, that you "
                "used a 16-char App Password (not your normal password), and "
                "that IMAP is enabled in Gmail settings."
            ) from exc

        # readonly=True -> selecting the mailbox never changes \\Seen flags.
        typ, _ = conn.select(folder, readonly=True)
        if typ != "OK":
            raise GmailError(f"Could not open folder {folder!r}.")

        criteria = "UNSEEN" if only_unread else "ALL"
        typ, data = conn.search(None, criteria)
        if typ != "OK":
            raise GmailError("Search failed on the mailbox.")

        ids = data[0].split()
        if not ids:
            return []
        # newest first, capped at `limit`
        chosen = ids[::-1][: max(1, int(limit))]

        results: list[InboxEmail] = []
        for msg_id in chosen:
            typ, msg_data = conn.fetch(msg_id, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)
            results.append(
                InboxEmail(
                    uid=msg_id.decode(errors="replace"),
                    sender=_decode(msg.get("From")),
                    subject=_decode(msg.get("Subject")) or "(no subject)",
                    date=_decode(msg.get("Date")),
                    body=_extract_body(msg),
                )
            )
        return results
    finally:
        try:
            conn.logout()
        except Exception:
            pass
