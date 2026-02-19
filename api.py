import os
from flask import Flask, request, jsonify
from mistralai import Mistral, UserMessage

app = Flask(__name__)

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY is not set")

client = Mistral(api_key=api_key)

def mistral_call(user_message: str, model: str = "mistral-large-latest") -> str:
    messages = [UserMessage(content=user_message)]
    resp = client.chat.complete(model=model, messages=messages)
    return resp.choices[0].message.content

CLASSIFY_PROMPT = """
You are a bank customer service bot.
Classify the inquiry into ONE of:
card arrival
change pin
exchange rate
country support
cancel transfer
charge dispute
customer service

Only return the category name.

Inquiry: {inquiry}
Category:
"""

def classify_inquiry(inquiry: str) -> str:
    out = mistral_call(CLASSIFY_PROMPT.format(inquiry=inquiry)).strip().lower()
    allowed = {
        "card arrival","change pin","exchange rate","country support",
        "cancel transfer","charge dispute","customer service"
    }
    for a in allowed:
        if a in out:
            return a
    return "customer service"

def generate_support_reply(inquiry: str, category: str) -> str:
    prompt = f"""
You are a professional bank customer support assistant.

Detected category: {category}

Customer inquiry:
{inquiry}

Write a helpful response in 3 to 6 lines.
"""
    return mistral_call(prompt).strip()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    category = classify_inquiry(message)
    reply = generate_support_reply(message, category)

    return jsonify({
        "message": message,
        "category": category,
        "reply": reply
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
