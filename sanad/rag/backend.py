import re
import json
from pathlib import Path
from openai import OpenAI
from rag.retriever import HybridRetriever


# ---------- Prepare ----------
# The knowledge base now travels with this package, so it is located relative to this file
# instead of the current working directory. Same file, same contents, same articles.
KNOWLEDGE_BASE = Path(__file__).resolve().parent / "data" / "labor_law" / "labor_law_parsed.json"
documents = json.load(open(KNOWLEDGE_BASE, encoding="utf-8"))
hybrid = HybridRetriever(documents)


# ---------- Utilities ----------
def detect_language(text: str) -> str:
    """Detect if the question is Arabic or English (robustly)."""
    if not text.strip():
        return "en"
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return "ar" if arabic_chars > len(text) / 2 else "en"


def highlight_articles(text: str):
    return re.sub(r"(المادة\s+[^\s،.]+|Article\s+\d+)", r"**\1**", text)


def get_retriever():
    return hybrid


# ---------- Core Answer ----------
def generate_answer(query: str, context: str, lang: str, api_key: str) -> str:
    """Generate an answer using a per-user OpenAI API key."""
    client = OpenAI(api_key=api_key)

    if lang == "ar":
        prompt = f"""أنت مساعد ذكي متخصص في نظام العمل السعودي.
اعتمد فقط على النصوص أدناه للإجابة على السؤال بدقة وبالعربية.

النصوص ذات الصلة:
\"\"\"{context}\"\"\"

السؤال: {query}

الإجابة:"""
    else:
        prompt = f"""You are an intelligent assistant specialized in Saudi Labor Law.
Use only the following text to answer accurately in English.

Relevant Articles:
\"\"\"{context}\"\"\"

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip()
    return highlight_articles(answer)


# ---------- Main Entry ----------
def answer_policy_question(query: str, employee_data: dict | None = None, api_key: str | None = None):
    """Answer a policy question, optionally using employee data and user-provided key."""
    if not api_key:
        raise ValueError("OpenAI API key is required for this session.")

    retriever = get_retriever()
    lang = detect_language(query)

    # Enrich query with employee info if provided
    if employee_data:
        info = "\n".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in employee_data.items()])
        query += f"\n\nEmployee Info:\n{info}"

    # Retrieve context
    results = retriever.retrieve(query)
    if not results:
        msg = "❌ لم يتم العثور على مواد ذات صلة." if lang == "ar" else "❌ No relevant articles found."
        return msg, []

    context = "\n\n".join(r["content"] for r in results)
    answer = generate_answer(query, context, lang, api_key)

    # Build references
    references = [{
        "similarity": round(r["score"], 3),
        "part": r["metadata"].get("part_title_ar", ""),
        "chapter": r["metadata"].get("chapter_title_ar", ""),
        "article_name": r["metadata"].get("arabic_name", "غير معروفة"),
        "article_number": r["metadata"].get("number_ar", ""),
        "arabic_content": r["metadata"].get("arabic_content", ""),
        "english_content": r["metadata"].get("english_content", "")
    } for r in results]


    return answer, references
