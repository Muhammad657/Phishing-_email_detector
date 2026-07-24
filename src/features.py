"""Feature extraction for the phishing email detector.

Two complementary views of an email are combined:

1. TF-IDF over the raw text  -> captures vocabulary / phrasing.
2. Hand-engineered numeric signals -> capture structural red flags that
   phishing emails share regardless of exact wording (lots of links, urgency
   language, requests for credentials, IP-address URLs, etc.).

`build_features()` returns a scikit-learn transformer that produces the
combined feature matrix, so it can be dropped straight into a Pipeline and
persisted alongside the trained model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

# --- regexes / word lists used by the engineered features -------------------

URL_RE = re.compile(r"https?://[^\s\"'>)]+", re.IGNORECASE)
# a "url" written without a scheme, e.g. www.paypa1-secure.com/login
BARE_DOMAIN_RE = re.compile(r"\b(?:www\.)?[a-z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?", re.IGNORECASE)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

URGENCY_WORDS = [
    "urgent", "immediately", "right away", "as soon as possible", "asap",
    "act now", "final notice", "last warning", "expire", "expires", "expiring",
    "suspend", "suspended", "suspension", "deactivat", "locked", "limited",
    "within 24 hours", "24 hours", "verify now", "action required",
]

CREDENTIAL_WORDS = [
    "password", "username", "login", "log in", "sign in", "credential",
    "ssn", "social security", "credit card", "card number", "cvv", "pin",
    "bank account", "account number", "routing number", "verify your account",
    "confirm your identity", "update your details", "billing information",
]

LURE_WORDS = [
    "congratulations", "you have won", "winner", "prize", "lottery",
    "free gift", "gift card", "claim your", "reward", "refund", "inheritance",
    "wire transfer", "bitcoin", "cryptocurrency", "investment opportunity",
]

GENERIC_GREETINGS = [
    "dear customer", "dear user", "dear account holder", "dear member",
    "dear client", "valued customer", "dear sir/madam", "hello dear",
]

# Brands commonly impersonated -- used to spot look-alike / mismatched domains.
KNOWN_BRANDS = [
    "paypal", "apple", "microsoft", "amazon", "google", "netflix", "facebook",
    "instagram", "bank", "chase", "wellsfargo", "americanexpress", "dhl",
    "fedex", "ups", "irs", "usps", "linkedin",
]

SUSPICIOUS_TLDS = {
    ".ru", ".cn", ".tk", ".top", ".xyz", ".click", ".zip", ".review",
    ".country", ".kim", ".work", ".party", ".gq", ".ml", ".cf", ".ga",
}


def _count_any(text: str, needles: list[str]) -> int:
    """How many of the phrases in `needles` appear in `text` (lowercased)."""
    return sum(text.count(n) for n in needles)


def _extract_urls(text: str) -> list[str]:
    return URL_RE.findall(text)


def engineered_features(text: str) -> dict[str, float]:
    """Compute the interpretable numeric signals for a single email string."""
    raw = text or ""
    low = raw.lower()
    urls = _extract_urls(raw)
    n_chars = max(len(raw), 1)

    # Look-alike domain: a known brand name appears in the body but the links
    # point at a domain that does NOT contain that brand -> classic spoof.
    link_domains = " ".join(urls).lower()
    brand_in_text = any(b in low for b in KNOWN_BRANDS)
    brand_in_links = any(b in link_domains for b in KNOWN_BRANDS)
    brand_mismatch = 1.0 if (brand_in_text and urls and not brand_in_links) else 0.0

    suspicious_tld = 1.0 if any(tld in link_domains for tld in SUSPICIOUS_TLDS) else 0.0

    # A domain with digits substituted for letters, e.g. paypa1, g00gle.
    has_lookalike = 1.0 if re.search(r"[a-z]+[0-9]+[a-z]*\.[a-z]{2,}", link_domains) else 0.0

    return {
        "n_urls": float(len(urls)),
        "n_ip_urls": float(len(IP_URL_RE.findall(raw))),
        "n_emails": float(len(EMAIL_RE.findall(raw))),
        "urgency": float(_count_any(low, URGENCY_WORDS)),
        "credential_request": float(_count_any(low, CREDENTIAL_WORDS)),
        "lure": float(_count_any(low, LURE_WORDS)),
        "generic_greeting": float(_count_any(low, GENERIC_GREETINGS)),
        "n_exclaim": float(raw.count("!")),
        "n_uppercase_words": float(sum(1 for w in raw.split() if len(w) > 1 and w.isupper())),
        "has_html": 1.0 if re.search(r"<[a-z][^>]*>", low) else 0.0,
        "money_symbols": float(len(re.findall(r"[$€£]|\busd\b|\beur\b", low))),
        "brand_mismatch": brand_mismatch,
        "suspicious_tld": suspicious_tld,
        "lookalike_domain": has_lookalike,
        "link_density": float(len(link_domains)) / n_chars,
        "length": float(n_chars),
    }


# Stable ordering of the engineered feature columns.
FEATURE_NAMES: list[str] = list(engineered_features("").keys())


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapping `engineered_features` for a list of emails."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        rows = [
            [feats[name] for name in FEATURE_NAMES]
            for feats in (engineered_features(t) for t in X)
        ]
        return csr_matrix(np.asarray(rows, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def build_features(max_features: int = 20000) -> FeatureUnion:
    """The combined (TF-IDF + engineered) feature transformer."""
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
    engineered = Pipeline([
        ("extract", EngineeredFeatures()),
        # scale so no single count dominates the linear model.
        ("scale", MaxAbsScaler()),
    ])
    return FeatureUnion([
        ("tfidf", tfidf),
        ("engineered", engineered),
    ])


@dataclass
class Explanation:
    """A human-readable reason contributing to a phishing verdict."""
    signal: str
    value: float


def explain(text: str) -> list[Explanation]:
    """Return the engineered signals that fired for an email (value > 0)."""
    feats = engineered_features(text)
    interesting = [
        "n_urls", "n_ip_urls", "urgency", "credential_request", "lure",
        "generic_greeting", "brand_mismatch", "suspicious_tld",
        "lookalike_domain", "n_exclaim",
    ]
    out = [Explanation(k, feats[k]) for k in interesting if feats[k] > 0]
    return sorted(out, key=lambda e: e.value, reverse=True)
