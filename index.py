"""Private Yandex Alice skill -> Google Gemini API for Vercel Python runtime.

No external packages. Never log secrets or user utterances.
"""
from http.server import BaseHTTPRequestHandler
import hmac
import json
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
PROMPT = (
    "Ты дружелюбный голосовой помощник на умной колонке. "
    "Отвечай по-русски ясно, коротко и естественно, обычно одним-четырьмя предложениями. "
    "Твой ответ произносится голосом: никаких Markdown, ссылок и длинных перечислений. "
    "Если вопрос требует актуальных сведений, но у тебя нет доступа к поиску, "
    "скажи, что нужно проверить дату или источник. "
    "Не утверждай, что управляешь умным домом, музыкой или выполняешь действия Алисы."
)
EXIT_WORDS = {"выход", "выйти", "закончить", "закрой навык", "стоп"}
RESET_WORDS = {"сброс", "новый разговор", "начать заново"}


def _voice_response(text, history=None, end=False):
    body = {
        "version": "1.0",
        "response": {"text": (text or "Не получилось ответить.")[:980], "end_session": end},
    }
    if not end:
        body["session_state"] = {"h": history or []}
    return body


def _history(pairs):
    """Store at most 2 short turns below Alice's 1-KiB session-state limit."""
    clean = []
    if isinstance(pairs, list):
        for pair in pairs[-2:]:
            if isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair):
                clean.append([pair[0][:160], pair[1][:220]])
    while clean and len(json.dumps({"h": clean}, ensure_ascii=False).encode("utf-8")) > 850:
        if len(clean) > 1:
            clean.pop(0)
        else:
            # Trim a single pair by bytes, preserving valid UTF-8.
            clean[0][1] = clean[0][1][:max(0, len(clean[0][1]) - 20)]
            if not clean[0][1]:
                clean = []
    return clean


def _clean_text(text):
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"(?m)\[[^\]]+\]\([^)]*\)", "", text)  # strip inline markdown links
    return text[:950].strip()


def _ask_gemini(utterance, history, api_key):
    contents = []
    for question, answer in history:
        contents.append({"role": "user", "parts": [{"text": question}]})
        contents.append({"role": "model", "parts": [{"text": answer}]})
    contents.append({"role": "user", "parts": [{"text": utterance[:1200]}]})
    payload = {
        "systemInstruction": {"parts": [{"text": PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 260, "thinkingConfig": {"thinkingLevel": "minimal"}},
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3.0) as response:
        data = json.loads(response.read(300_000).decode("utf-8"))
    parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    answer = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))
    return _clean_text(answer)


def process_alice(body, api_key, skill_id):
    session = body.get("session") or {}
    if skill_id and session.get("skill_id") != skill_id:
        return 403, _voice_response("Навык вызван из другого проекта.")
    request = body.get("request") or {}
    utterance = str(request.get("original_utterance") or request.get("command") or "").strip()
    simplified = utterance.casefold().rstrip(".!? ")
    history = _history((body.get("state") or {}).get("session", {}).get("h", []))

    if simplified in EXIT_WORDS:
        return 200, _voice_response("До встречи!", end=True)
    if simplified in RESET_WORDS:
        return 200, _voice_response("Начнём новый разговор. Что хотите узнать?", history=[])
    if not utterance or (session.get("new") and simplified in {"умный собеседник", "мой умный собеседник"}):
        return 200, _voice_response("Привет! Это ваш помощник Gemini. Задайте вопрос.", history=[])
    if not api_key:
        return 200, _voice_response("Ключ Gemini пока не настроен в Vercel.", history=history)

    try:
        answer = _ask_gemini(utterance, history, api_key)
        if not answer:
            return 200, _voice_response("Пока не получилось сформировать ответ. Спросите иначе.", history=history)
        history = _history(history + [[utterance[:160], answer[:220]]])
        return 200, _voice_response(answer, history=history)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            phrase = "Бесплатный лимит Gemini временно исчерпан. Попробуйте позднее."
        elif exc.code in (401, 403):
            phrase = "Google отклонил запрос. Проверьте доступ к Gemini API и ключ."
        elif exc.code == 404:
            phrase = "Выбранная модель Gemini недоступна для этого проекта."
        else:
            phrase = "Gemini сейчас не отвечает. Попробуйте ещё раз."
        return 200, _voice_response(phrase, history=history)
    except (TimeoutError, socket.timeout, urllib.error.URLError):
        return 200, _voice_response("Gemini отвечает слишком долго. Повторите короткий вопрос.", history=history)
    except (ValueError, KeyError, IndexError, TypeError):
        return 200, _voice_response("Не удалось прочитать ответ Gemini. Попробуйте снова.", history=history)


class handler(BaseHTTPRequestHandler):
    def _send(self, status, result):
        data = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, {"status": "ok", "service": "alice-gemini"})

    def do_POST(self):
        token = (parse_qs(urlsplit(self.path).query).get("token") or [""])[0]
        expected = os.environ.get("ALICE_WEBHOOK_TOKEN", "")
        if not expected or not hmac.compare_digest(token, expected):
            self._send(403, {"error": "Forbidden"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not (0 < length <= 32_000):
                raise ValueError("Invalid length")
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(body, dict):
                raise ValueError("Invalid body")
        except (ValueError, UnicodeError, json.JSONDecodeError):
            self._send(400, {"error": "Invalid JSON"})
            return
        status, result = process_alice(
            body,
            os.environ.get("GEMINI_API_KEY", ""),
            os.environ.get("ALICE_SKILL_ID", ""),
        )
        self._send(status, result)

    def log_message(self, format, *args):
        # Do not log query-string secrets, requests or user utterances.
        return
