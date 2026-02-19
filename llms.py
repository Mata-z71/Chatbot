import os
import json
import streamlit as st
from mistralai import Mistral, UserMessage

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="LLMs Lab - Mistral", page_icon="🤖", layout="wide")
st.title("🤖 LLMs Lab Web App (Mistral)")
st.caption("Includes: Chatbot, Classification, JSON Extraction, Email Response, Summarization")

# -----------------------------
# API Key check
# -----------------------------
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    st.error(
        "MISTRAL_API_KEY is not set.\n\n"
        "✅ Windows PowerShell:\n"
        "1) $env:MISTRAL_API_KEY=\"YOUR_KEY\"\n"
        "2) streamlit run LLMs.py\n\n"
        "Make sure you run both commands in the SAME terminal window."
    )
    st.stop()

# -----------------------------
# Mistral client (cache so it doesn't recreate each rerun)
# -----------------------------
@st.cache_resource
def get_client(key: str):
    return Mistral(api_key=key)

client = get_client(api_key)

def mistral_call(user_message: str, model: str = "mistral-large-latest") -> str:
    """Basic call to Mistral chat completion."""
    messages = [UserMessage(content=user_message)]
    resp = client.chat.complete(model=model, messages=messages)
    return resp.choices[0].message.content

def safe_json_parse(text: str):
    """
    Tries to parse JSON even if the model wraps it in ```json ... ```
    Returns (data, error_message)
    """
    cleaned = text.strip()

    # Remove markdown fences if present
    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(cleaned), None
    except Exception as e:
        return None, str(e)

# ============================================================
# PROMPTS (from your lab)
# ============================================================

CLASSIFY_PROMPT = """
You are a bank customer service bot.
Your task is to assess customer intent and categorize customer
inquiry after <<<>>> into one of the following predefined categories:
card arrival
change pin
exchange rate
country support
cancel transfer
charge dispute
If the text doesn't fit into any of the above categories,
classify it as:
customer service
You will only respond with the predefined category.
Do not provide explanations or notes.
###
Here are some examples:
Inquiry: How do I know if I will get my card, or if it is lost? I am concerned about the delivery process and would like to ensure that I will receive my card.
Category: card arrival
Inquiry: I am planning an international trip to Paris and would like to inquire about the current exchange rates for Euros as well as any associated fees for currency exchange.
Category: exchange rate
Inquiry: What countries are getting support? I will be traveling and living abroad for an extended period of time, specifically in France and Germany, and want to know if your cards will work there.
Category: country support
Inquiry: Can I get help starting my computer? I am having difficulty starting my computer.
Category: customer service
###
<<<
Inquiry: {inquiry}
>>>
Category:
"""

MORTGAGE_FACTS = """# Facts
30-year fixed-rate: interest rate 6.403%, APR 6.484%
20-year fixed-rate: interest rate 6.329%, APR 6.429%
15-year fixed-rate: interest rate 5.705%, APR 5.848%
10-year fixed-rate: interest rate 5.500%, APR 5.720%
7-year ARM: interest rate 7.011%, APR 7.660%
5-year ARM: interest rate 6.880%, APR 7.754%
3-year ARM: interest rate 6.125%, APR 7.204%
30-year fixed-rate FHA: interest rate 5.527%, APR 6.316%
30-year fixed-rate VA: interest rate 5.684%, APR 6.062%
"""

# ============================================================
# FUNCTIONS for each feature
# ============================================================

def classify_inquiry(inquiry: str) -> str:
    prompt = CLASSIFY_PROMPT.format(inquiry=inquiry)
    out = mistral_call(prompt).strip().lower()

    allowed = {
        "card arrival",
        "change pin",
        "exchange rate",
        "country support",
        "cancel transfer",
        "charge dispute",
        "customer service",
    }
    # Normalize output in case model adds extra text
    for a in allowed:
        if a in out:
            return a
    return "customer service"

def bank_support_reply(inquiry: str, category: str) -> str:
    prompt = f"""
You are a professional bank customer support assistant.

Detected category: {category}

Customer inquiry:
{inquiry}

Write a helpful response:
- friendly & professional
- 3 to 6 lines
- ask for missing info only if necessary
- do NOT mention internal prompts or that you are an AI
"""
    return mistral_call(prompt).strip()

def extract_medical_json(notes: str) -> tuple[str, dict | None, str | None]:
    prompt = f"""
Extract information from the following medical notes:
{notes}

Return ONLY valid JSON with this schema:
{{
  "age": {{"type":"integer"}},
  "gender": {{"type":"string","enum":["male","female","other"]}},
  "diagnosis": {{"type":"string","enum":["migraine","diabetes","arthritis","acne"]}},
  "weight": {{"type":"integer"}},
  "smoking": {{"type":"string","enum":["yes","no"]}}
}}
"""
    raw = mistral_call(prompt).strip()
    data, err = safe_json_parse(raw)
    return raw, data, err

def generate_mortgage_email(email_text: str) -> str:
    prompt = f"""
You are a mortgage lender customer service bot, and your task is to
create personalized email responses to address customer questions.
Answer the customer's inquiry using the provided facts below. Ensure
that your response is clear, concise, and directly addresses the
customer's question. Address the customer in a friendly and
professional manner. Sign the email with "Lender Customer Support."

{MORTGAGE_FACTS}

# Email
{email_text}
"""
    return mistral_call(prompt).strip()

def summarize_newsletter(newsletter: str) -> str:
    prompt = f"""
You are a commentator. Your task is to write a report on a newsletter.
When presented with the newsletter, come up with interesting questions to ask,
and answer each question.
Afterward, combine all the information and write a report in the markdown
format.

# Newsletter:
{newsletter}

# Instructions:
## Summarize:
In clear and concise language, summarize the key points and themes
presented in the newsletter.
## Interesting Questions:
Generate three distinct and thought-provoking questions that can be
asked about the content of the newsletter. For each question:
- After "Q: ", describe the problem
- After "A: ", provide a detailed explanation of the problem addressed
in the question.
- Enclose the ultimate answer in <>.
## Write an analysis report
Using the summary and the answers to the interesting questions,
create a comprehensive report in Markdown format.
"""
    return mistral_call(prompt).strip()

# ============================================================
# UI
# ============================================================

with st.sidebar:
    st.header("✅ Lab Features")
    st.write("- Customer Support Chatbot")
    st.write("- Classification")
    st.write("- JSON Information Extraction")
    st.write("- Personalized Email Response")
    st.write("- Summarization Report")
    st.divider()
    show_debug = st.checkbox("Show debug (category + prompts)", value=False)

tabs = st.tabs([
    "1) Customer Support Chatbot",
    "2) Classification",
    "3) JSON Extraction",
    "4) Personalized Email",
    "5) Summarization"
])

# -----------------------------
# TAB 1: Customer Support Chatbot
# -----------------------------
with tabs[0]:
    st.subheader("Customer Support Chatbot")
    st.write("Type any bank support question. The bot classifies it, then replies.")

    examples = [
        "My card still hasn’t arrived. What should I do?",
        "I forgot my PIN. How can I change it?",
        "What is the exchange rate for EUR today?",
        "Do you support using the card in France and Germany?",
        "Please cancel my international transfer.",
        "I see a charge I don't recognize. How do I dispute it?"
    ]
    ex = st.selectbox("Example questions (optional)", ["(none)"] + examples)

    user_msg = st.text_area("Your message", value="" if ex == "(none)" else ex, height=120)

    if st.button("Send (Chatbot)", type="primary"):
        if not user_msg.strip():
            st.warning("Please type a message.")
        else:
            try:
                with st.spinner("Classifying + generating reply..."):
                    cat = classify_inquiry(user_msg)
                    reply = bank_support_reply(user_msg, cat)

                st.success("Done!")
                st.markdown("### Detected category")
                st.write(cat)

                st.markdown("### Reply")
                st.write(reply)

                if show_debug:
                    st.markdown("### Debug")
                    st.code(CLASSIFY_PROMPT)

            except Exception as e:
                st.error("Error calling Mistral.")
                st.code(str(e))

# -----------------------------
# TAB 2: Classification (standalone)
# -----------------------------
with tabs[1]:
    st.subheader("Classification Task")
    inquiry = st.text_area("Enter an inquiry to classify", height=120,
                           placeholder="e.g., I am inquiring about the availability of your cards in the EU")

    if st.button("Classify", type="primary"):
        if not inquiry.strip():
            st.warning("Please enter an inquiry.")
        else:
            try:
                with st.spinner("Classifying..."):
                    cat = classify_inquiry(inquiry)
                st.markdown("### Category")
                st.write(cat)
            except Exception as e:
                st.error("Error calling Mistral.")
                st.code(str(e))

# -----------------------------
# TAB 3: JSON Extraction
# -----------------------------
with tabs[2]:
    st.subheader("Information Extraction (JSON)")
    default_notes = """A 60-year-old male patient, Mr. Johnson, presented with symptoms
of increased thirst, frequent urination, fatigue, and unexplained
weight loss. Upon evaluation, he was diagnosed with diabetes,
confirmed by elevated blood sugar levels. Mr. Johnson's weight
is 210 lbs. He has been prescribed Metformin to be taken twice daily
with meals. It was noted during the consultation that the patient is
a current smoker."""
    notes = st.text_area("Medical notes", value=default_notes, height=180)

    if st.button("Extract JSON", type="primary"):
        if not notes.strip():
            st.warning("Please enter notes.")
        else:
            try:
                with st.spinner("Extracting..."):
                    raw, data, err = extract_medical_json(notes)

                st.markdown("### Model output (raw)")
                st.code(raw)

                if data is not None:
                    st.markdown("### Parsed JSON")
                    st.json(data)
                else:
                    st.warning("Could not parse JSON. The model output may not be valid JSON.")
                    st.code(err)

            except Exception as e:
                st.error("Error calling Mistral.")
                st.code(str(e))

# -----------------------------
# TAB 4: Personalized Email
# -----------------------------
with tabs[3]:
    st.subheader("Personalized Email Response (Mortgage)")
    default_email = """Dear mortgage lender,
What's your 30-year fixed-rate APR, how is it compared to the 15-year
fixed rate?
Regards,
Anna"""
    email_text = st.text_area("Customer email", value=default_email, height=160)

    if st.button("Generate Email Reply", type="primary"):
        if not email_text.strip():
            st.warning("Please enter an email.")
        else:
            try:
                with st.spinner("Generating reply..."):
                    reply = generate_mortgage_email(email_text)

                st.markdown("### Email reply")
                st.write(reply)

            except Exception as e:
                st.error("Error calling Mistral.")
                st.code(str(e))

# -----------------------------
# TAB 5: Summarization
# -----------------------------
with tabs[4]:
    st.subheader("Summarization Task (Newsletter → Report)")
    default_newsletter = """European AI champion Mistral AI unveiled new large language models and formed an alliance with Microsoft.
What’s new: Mistral AI introduced two closed models, Mistral Large and Mistral Small (joining Mistral Medium, which debuted quietly late last year).
Model specs: The new models’ parameter counts, architectures, and training methods are undisclosed.
Mistral Large achieved 81.2 percent on the MMLU benchmark, outperforming Anthropic’s Claude 2, Google’s Gemini Pro, and Meta’s Llama 2 70B, though falling short of GPT-4.
Both models are fluent in French, German, Spanish, and Italian. They’re trained for function calling and JSON-format output.
Microsoft’s investment in Mistral AI is significant but tiny compared to its $13 billion stake in OpenAI.
Behind the news: Mistral AI was founded in early 2023 by engineers from Google and Meta.
Yes, but: Mistral AI’s partnership with Microsoft has divided European lawmakers and regulators.
Why it matters: The partnership gives the startup processing power and access to customers.
We’re thinking: Mistral AI has made impressive progress in a short time."""
    newsletter = st.text_area("Paste newsletter/article", value=default_newsletter, height=220)

    if st.button("Summarize + Q&A Report", type="primary"):
        if not newsletter.strip():
            st.warning("Please enter text.")
        else:
            try:
                with st.spinner("Writing report..."):
                    report_md = summarize_newsletter(newsletter)

                st.markdown("### Output report (Markdown)")
                st.markdown(report_md)

                st.download_button(
                    "Download report as .md",
                    data=report_md.encode("utf-8"),
                    file_name="summary_report.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error("Error calling Mistral.")
                st.code(str(e))
