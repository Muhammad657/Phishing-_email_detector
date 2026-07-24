import requests
import json
import re
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")


def extract_links(email_text):
    """Find all URLs in the email text."""
    url_pattern = r'https?://[^\s<>"\']+'
    return re.findall(url_pattern, email_text)


def check_link_destination(url):
    """Follows redirects (even through shorteners) and returns the final real destination."""
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        return {
            "original": url,
            "final_destination": response.url,
            "num_redirects": len(response.history)
        }
    except requests.RequestException as e:
        return {
            "original": url,
            "final_destination": None,
            "error": str(e)
        }


def explain_phishing_result(email_text, is_phishing, confidence=None):
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not set"}

    verdict = "PHISHING" if is_phishing else "SAFE"
    confidence_text = f" (confidence: {confidence*100:.0f}%)" if confidence else ""

    links = extract_links(email_text)
    link_info = [check_link_destination(link) for link in links]

    link_section = ""
    if link_info:
        link_section = "\n\nLinks found in this email (with their REAL final destination after following redirects):\n"
        for link in link_info:
            if link["final_destination"]:
                link_section += f'- Displayed/original: {link["original"]} → Actually goes to: {link["final_destination"]} ({link["num_redirects"]} redirect(s))\n'
            else:
                link_section += f'- Displayed/original: {link["original"]} → Could not resolve (error: {link.get("error")})\n'

    prompt = f"""An email classifier flagged this email as {verdict}{confidence_text}.

Email content:
\"\"\"
{email_text}
\"\"\"{link_section}

Respond with ONLY a JSON object (no other text, no markdown formatting) in exactly this format:
{{
  "verdict": "{verdict}",
  "summary": "one short sentence overall summary",
  "red_flags": ["short flag 1", "short flag 2", "short flag 3"],
  "flagged_keywords": ["specific word or phrase 1", "specific word or phrase 2"],
  "link_warning": "if any link's real destination looks suspicious or mismatched, describe it here in one sentence, otherwise null",
  "explanation": "2-3 sentence plain-English explanation of the reasoning"
}}

For "flagged_keywords", pull out the EXACT words/phrases from the email itself that suggest urgency, threats, credential requests, or manipulation (e.g. "verify your account", "act now", "suspended"). If there are no red flags (email is safe), return empty lists/null. Keep everything concise and non-technical."""

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            },
            timeout=15
        )
        data = response.json()
        raw_text = data["choices"][0]["message"]["content"]
    except Exception as e:
        return {"error": f"AI explanation failed: {e}"}

    clean_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        result = json.loads(clean_text)
    except json.JSONDecodeError:
        result = {
            "verdict": verdict,
            "summary": "Could not parse explanation",
            "red_flags": [],
            "flagged_keywords": [],
            "link_warning": None,
            "explanation": raw_text
        }

    result["link_details"] = link_info
    return result