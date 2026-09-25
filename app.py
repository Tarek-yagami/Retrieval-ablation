"""Streamlit demo: the ablation dashboard (headline result) plus a query
playground where retrieval actually runs live. Heavy computation (the
ablation itself, judged-answer scoring) happens offline in experiments/ and
is only read here from cached JSON; the playground indexes one small
dataset (scifact) live so it stays fast enough to deploy for free.

Run locally:
    streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from hybridrag.data.beir_loader import load_beir_dataset
from hybridrag.generation import generate_answer
from hybridrag.judge import judge_answer
from hybridrag.pipeline import CONFIGS, RetrievalPipeline

st.set_page_config(page_title="Retrieval Ablation", layout="wide")

ABLATION_RESULTS = Path("experiments/retrieval_ablation/results/results.json")
ANSWER_QUALITY_RESULTS = Path("experiments/answer_quality/results/results.json")
PLAYGROUND_DATASET = "scifact"

# Fixed categorical color per config, in a colorblind-safe order (Okabe-Ito),
# reused identically across every chart so a config's color never shifts.
CONFIG_COLORS = {
    "sparse": "#E69F00",
    "dense": "#56B4E9",
    "hybrid": "#009E73",
    "hybrid_rerank": "#CC79A7",
    "dense_rerank": "#0072B2",
    "sparse_rerank": "#D55E00",
}
METRICS = [("ndcg@10", "nDCG@10"), ("recall@100", "Recall@100"), ("mrr@10", "MRR@10")]


@st.cache_resource
def get_pipeline(dataset_name: str) -> tuple[RetrievalPipeline, dict]:
    dataset = load_beir_dataset(dataset_name)
    return RetrievalPipeline(dataset.corpus), dataset


def render_ablation_page() -> None:
    st.header("Retrieval ablation: does hybrid + rerank actually help?")

    if not ABLATION_RESULTS.exists():
        st.warning(
            f"No results yet. Run `python {ABLATION_RESULTS.parents[1] / 'run.py'}` first."
        )
        return

    results = json.loads(ABLATION_RESULTS.read_text())
    datasets = list(results.keys())

    cols = st.columns(len(METRICS))
    for col, (metric_key, metric_label) in zip(cols, METRICS):
        fig = go.Figure()
        for config in CONFIGS:
            fig.add_bar(
                name=config,
                x=datasets,
                y=[results[d][config][metric_key] for d in datasets],
                marker_color=CONFIG_COLORS[config],
                legendgroup=config,
                showlegend=True,
            )
        fig.update_layout(
            title=metric_label,
            barmode="group",
            yaxis_range=[0, 1],
            margin=dict(t=40, b=10, l=10, r=10),
            height=360,
        )
        col.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw numbers"):
        st.json(results)

    if ANSWER_QUALITY_RESULTS.exists():
        st.subheader("Answer quality: does better retrieval judge better too?")
        records = json.loads(ANSWER_QUALITY_RESULTS.read_text())
        by_config: dict[str, list[dict]] = {}
        for r in records:
            by_config.setdefault(r["config"], []).append(r)

        aq_cols = st.columns(len(by_config))
        for col, (config, rows) in zip(aq_cols, by_config.items()):
            n = len(rows)
            mean_faith = sum(r["faithfulness"] for r in rows) / n
            mean_rel = sum(r["relevance"] for r in rows) / n
            col.metric(f"{config} faithfulness", f"{mean_faith:.2f} / 5")
            col.metric(f"{config} relevance", f"{mean_rel:.2f} / 5")
    else:
        st.info(
            f"No answer-quality results yet. Run "
            f"`python {ANSWER_QUALITY_RESULTS.parents[1] / 'run.py'}` first."
        )


def render_playground_page() -> None:
    st.header(f"Query playground: live retrieval on `{PLAYGROUND_DATASET}`")

    pipeline, dataset = get_pipeline(PLAYGROUND_DATASET)
    preset_ids = sorted(dataset.queries.keys())[:25]
    preset_labels = {qid: dataset.queries[qid] for qid in preset_ids}

    choice = st.selectbox(
        "Pick a sample query, or choose 'custom' to type your own",
        options=["custom", *preset_ids],
        format_func=lambda qid: "custom (type below)" if qid == "custom" else preset_labels[qid],
    )
    query = st.text_input("Query", value="" if choice == "custom" else preset_labels[choice])
    if not query:
        st.stop()

    st.subheader("Retrieved passages per config")
    cols = st.columns(len(CONFIGS))
    top_passages: dict[str, list[str]] = {}
    for col, config in zip(cols, CONFIGS):
        col.markdown(f"**{config}**")
        hits = pipeline.search(query, config, top_k=3)
        passages = []
        for doc_id, score in hits:
            text = dataset.corpus[doc_id].get("text", "")[:300]
            passages.append(text)
            col.caption(f"score {score:.3f}")
            col.write(text + ("..." if len(text) == 300 else ""))
        top_passages[config] = passages

    st.divider()
    st.subheader("Generate + judge an answer")
    st.caption(
        "Uses the best-performing config (dense_rerank) and calls a live LLM "
        "via HYBRIDRAG_MODEL, off by default so browsing this demo never spends API credits."
    )
    if st.checkbox("Generate answer now"):
        with st.spinner("Generating..."):
            passages = top_passages["dense_rerank"]
            answer = generate_answer(query, passages)
            score = judge_answer(query, answer, passages)

        st.write(answer)
        c1, c2 = st.columns(2)
        c1.metric("Faithfulness", f"{score.faithfulness} / 5")
        c2.metric("Relevance", f"{score.relevance} / 5")
        st.caption(score.reasoning)


page = st.sidebar.radio("View", ["Ablation results", "Query playground"])
if page == "Ablation results":
    render_ablation_page()
else:
    render_playground_page()
