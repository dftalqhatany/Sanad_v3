"""Sanad user interface (Streamlit), in the style of the existing hr_assistant app.

It talks to the Sanad API over HTTP and nothing else: no agent, orchestrator, parser or RAG import,
and no routing or analysis logic. Run it with:

    python -m sanad.api                      # backend  (http://127.0.0.1:8000)
    streamlit run frontend/app.py            # this UI  (http://localhost:8501)
"""

from __future__ import annotations

import json
import os

import streamlit as st

from frontend.client import DEFAULT_BASE_URL, SanadApiClient
from frontend import view

st.set_page_config(page_title="Sanad — Employment Contract Assistant", layout="wide")
TASKS = {
    "auto": "Decide automatically / تلقائي",
    "contract_analysis": "Analyse one contract",
    "contract_comparison": "Compare contracts",
    "cv_analysis": "Analyse a CV",
    "regulatory_question": "Ask the labor law",
    "salary_benchmark": "Salary benchmarking",
}
ROLES = {"auto": "Detect", "contract": "Contract", "cv": "CV"}


# --------------------------------------------------------------------------- sidebar
st.sidebar.header("⚙️ Sanad API")
base_url = st.sidebar.text_input("API address", value=os.environ.get("SANAD_API_URL", DEFAULT_BASE_URL))
client = SanadApiClient(base_url=base_url)
config_response = client.config()
if not config_response.ok:
    st.sidebar.error(config_response.message)
    config = {"allowed_file_types": [".pdf", ".docx"], "max_files": 6, "max_contracts": 5, "max_file_size_mb": 20,
              "comparison_priorities": [], "answer_generation_enabled": False}
else:
    config = config_response.result
    st.sidebar.success("Connected")
st.sidebar.caption(f"Accepted files: {', '.join(config['allowed_file_types'])} · up to "
                   f"{config['max_file_size_mb']:g} MB each · {config['max_contracts']} contracts at most")
if not config.get("answer_generation_enabled"):
    st.sidebar.info("No OPENAI_API_KEY on the server: questions return retrieved articles without a written answer, "
                    "and contract findings stay 'requires review'.")

st.title("Sanad — Employment Contract Assistant")
st.caption("Upload employment contracts (PDF or DOCX), optionally a CV, or ask about the Saudi Labor Law. "
           "Sanad routes the request to its agents and shows the evidence behind every finding.")

# --------------------------------------------------------------------------- request form
left, right = st.columns([2, 1])
with left:
    uploaded = st.file_uploader(f"Contracts and CV ({', '.join(config['allowed_file_types'])})",
                                type=[t.lstrip(".") for t in config["allowed_file_types"]], accept_multiple_files=True)
    question = st.text_area("Regulatory question (optional) / سؤال نظامي", height=80,
                            placeholder="What is the maximum probation period? / ما هي المدة القصوى لفترة التجربة؟")
with right:
    task = st.selectbox("Task", options=list(TASKS), format_func=lambda key: TASKS[key])
    priorities = st.multiselect("Comparison priorities (in order)", options=config.get("comparison_priorities", []),
                                help="Applied one after another when comparing contracts.")
    job_title = st.text_input("Target job title (optional)", help="Compares the CV with this job instead of the "
                                                                  "contract's job title.")
    required_skills = st.text_input("Required skills (comma separated, optional)")
    salary_city = st.text_input("City for salary benchmarking (optional)",
                                help="Used with the job title to search the documented salary sources.")
    years_experience = st.text_input("Years of experience (optional)")

roles: list[str] = []
labels: list[str] = []
if uploaded:
    st.write("**Uploaded files**")
    for position, item in enumerate(uploaded, 1):
        columns = st.columns([3, 1, 2])
        columns[0].write(f"`{item.name}` — {len(item.getvalue()) / 1024:.0f} KB")
        roles.append(columns[1].selectbox("Role", options=list(ROLES), format_func=lambda key: ROLES[key],
                                          key=f"role_{position}", label_visibility="collapsed"))
        labels.append(columns[2].text_input("Label", value="", key=f"label_{position}",
                                            placeholder="Label (optional)", label_visibility="collapsed"))

submitted = st.button("Analyse", type="primary")


def target_job_json() -> str | None:
    if not job_title.strip():
        return None
    skills = [skill.strip() for skill in required_skills.split(",") if skill.strip()]
    return json.dumps({"title": job_title.strip(), "required_skills": skills})


# --------------------------------------------------------------------------- submit
if submitted:
    if not uploaded and not question.strip():
        st.warning("Upload a contract or CV, or type a regulatory question.")
        st.stop()
    with st.spinner("Sanad is working…"):
        if not uploaded and question.strip() and task in ("auto", "regulatory_question"):
            response = client.ask(question.strip())
        else:
            uploads = [(item.name, item.getvalue(), roles[position]) for position, item in enumerate(uploaded)]
            response = client.analyze(uploads, question=question.strip() or None, task=task, priorities=priorities,
                                      target_job_json=target_job_json(), labels=labels,
                                      salary={"salary_location": salary_city, "years_experience": years_experience,
                                              "salary_job_title": job_title})
    st.session_state["response"] = response

response = st.session_state.get("response")
if response is None:
    st.stop()
if not response.ok and response.result is None:
    st.error(f"{response.message}")
    st.stop()

result = response.result
summary = view.overview(result)

# --------------------------------------------------------------------------- result
badge = {"success": st.success, "partial": st.warning, "insufficient_evidence": st.warning,
         "invalid_input": st.error, "rag_error": st.error, "analysis_error": st.error}
badge.get(summary["status"], st.info)(f"**{summary['status_label']}** — {summary['summary']}")
st.caption(f"Routed to {summary['route_label']} ({summary['agent']}), rule: {summary['rule']}. {summary['reason']}")
for message in summary["errors"]:
    st.error(message)

documents = view.document_rows(result)
if documents:
    with st.expander(f"Uploaded documents ({len(documents)})", expanded=summary["status"] != "success"):
        st.dataframe(documents, use_container_width=True, hide_index=True)

answer = view.regulatory_answer(result)
if answer:
    st.subheader("Saudi Labor Law")
    if answer["answer"]:
        st.write(answer["answer"])
    if answer["message"]:
        st.info(answer["message"])
    for item in answer["evidence"]:
        with st.expander(f"{item['citation']} (score {item['score']:.3f})"):
            st.markdown(f"**العربية**\n\n{item['arabic_text']}")
            if item["english_text"]:
                st.markdown(f"**English (machine translation)**\n\n{item['english_text']}")

analysis = view.contract_analysis(result)
if analysis:
    st.subheader("Contract analysis")
    st.write(analysis["overall_summary"])
    st.dataframe(view.finding_rows(analysis), use_container_width=True, hide_index=True)
    evidence = view.evidence_items(analysis)
    if evidence:
        with st.expander(f"Regulatory evidence ({len(evidence)} articles retrieved)"):
            for item in evidence:
                st.markdown(f"**{item['citation']}** — rank {item['rank']}, score {item['score']:.3f}")
                st.markdown(item["arabic_text"] or "")
                if item["english_text"]:
                    st.caption(item["english_text"])
                st.divider()

cv = view.cv_analysis(result)
if cv:
    st.subheader("CV")
    st.write(cv["overall_summary"])
    rows = view.compatibility_rows(cv)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    observations = [o["message"] for o in cv.get("observations", [])]
    if observations:
        with st.expander(f"CV observations ({len(observations)})"):
            for observation in observations:
                st.write("• " + observation)

comparison = result.get("comparison")
if comparison:
    st.subheader("Contract comparison")
    recommendation = view.recommendation_view(comparison)
    if recommendation["status"] == "preferred_contract":
        st.success(f"**Preferred: {recommendation['preferred']}** — {recommendation['explanation']}")
    else:
        st.info(recommendation["explanation"])
    if recommendation["factors"]:
        st.markdown("**Why**")
        st.dataframe(recommendation["factors"], use_container_width=True, hide_index=True)
    for trade_off in recommendation["trade_offs"]:
        st.write("• " + trade_off)
    st.caption(recommendation["basis"])
    st.markdown("**Compared terms**")
    st.dataframe(view.comparison_rows(comparison), use_container_width=True, hide_index=True)
    risks = view.risk_rows(comparison)
    if risks:
        st.markdown("**Risks and review items**")
        st.dataframe(risks, use_container_width=True, hide_index=True)

salary = view.salary_view(result)
if salary:
    st.subheader("Salary")
    columns = st.columns(4)
    columns[0].metric("Contract salary", salary["contract_salary"])
    columns[1].metric(f"Market range ({salary['period']})", salary["market_range"])
    columns[2].metric("Location", salary["location"])
    columns[3].metric("Evidence", salary["data_quality"])
    if salary["position"]:
        st.write(f"The contract salary is **{salary['position']}** ({salary['basis']}).")
    st.caption(salary["message"])
    if salary["query"]:
        st.caption(f"Searched for: `{salary['query']}`")
    for source in salary["sources"]:
        st.write(f"• [{source['name']}]({source['url']}) — {source['tier']} source, retrieved {source['retrieved_at']}")
    if salary["evidence"]:
        with st.expander(f"Salary evidence ({len(salary['evidence'])} figures read from the sources)"):
            st.dataframe(salary["evidence"], use_container_width=True, hide_index=True)
    for limitation in salary["limitations"]:
        st.caption("• " + limitation)

notes = view.caveats(result)
if notes:
    with st.expander(f"Caveats and warnings ({len(notes)})", expanded=False):
        for note in notes:
            st.write("• " + note)

with st.expander("Raw result (JSON)"):
    st.json(result)
st.caption(summary["disclaimer"])
