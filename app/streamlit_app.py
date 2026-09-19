"""
Tourism Experience Analytics -- Streamlit application.

Five pages:
  Overview        headline numbers and model scorecard
  Explore         EDA figures and the SQL analysis layer
  Rating          predict the rating a visitor will give an attraction
  Visit mode      predict how a visitor is travelling
  Recommend       personalised attraction suggestions

Run with:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLEANING_LOG,
    CLF_MODEL_PATH,
    EDA_REPORT,
    FIGURE_DIR,
    MASTER_PARQUET,
    METRICS_PATH,
    RECO_MODEL_PATH,
    REG_MODEL_PATH,
    SQLITE_DB,
    TOP_N_RECOMMENDATIONS,
)

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
#
# Palette: deep petrol ink on a cool off-white, with a single teal accent.
# Amber is reserved exclusively for the "needs attention" state, so colour
# carries meaning rather than decoration.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fraunces:opsz,wght@9..144,500;9..144,600&display=swap');

      html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
      h1, h2, h3 { font-family: 'Fraunces', Georgia, serif; letter-spacing: -0.01em; }
      h1 { font-size: 2.1rem; margin-bottom: 0.2rem; }

      .lede { color: #5A6B78; font-size: 1.02rem; margin-bottom: 1.4rem; max-width: 62ch; }

      .metric-card {
        background: #FFFFFF; border: 1px solid #DCE4E9; border-left: 3px solid #0E7C7B;
        border-radius: 3px; padding: 0.9rem 1.1rem;
      }
      .metric-card .value { font-size: 1.75rem; font-weight: 600; color: #16323F; line-height: 1.1; }
      .metric-card .label { font-size: 0.8rem; color: #5A6B78; margin-top: 0.15rem; }
      .metric-card.attention { border-left-color: #C86A1E; }

      .note {
        background: #EFF3F5; border-left: 3px solid #7A8B99;
        padding: 0.75rem 1rem; border-radius: 3px; font-size: 0.9rem; color: #33505E;
      }
      div[data-testid="stSidebarNav"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_master() -> pd.DataFrame | None:
    if not MASTER_PARQUET.exists():
        return None
    return pd.read_parquet(MASTER_PARQUET)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_regressor():
    if not REG_MODEL_PATH.exists():
        return None
    from src.models import regression
    return regression.load_model()


@st.cache_resource(show_spinner=False)
def load_classifier():
    if not CLF_MODEL_PATH.exists():
        return None
    from src.models import classification
    return classification.load_model()


@st.cache_resource(show_spinner=False)
def load_recommender():
    if not RECO_MODEL_PATH.exists():
        return None
    from src.models import recommender
    return recommender.load_model()


@st.cache_data(show_spinner=False)
def run_sql(name: str) -> pd.DataFrame:
    from src.database import run_query
    return run_query(name)


def metric_card(value, label, attention: bool = False) -> str:
    cls = "metric-card attention" if attention else "metric-card"
    return f'<div class="{cls}"><div class="value">{value}</div><div class="label">{label}</div></div>'


def pipeline_missing() -> None:
    st.error("No processed data found.")
    st.markdown(
        "Build the dataset and models first:\n\n"
        "```bash\n"
        "python scripts/generate_sample_data.py   # or place the real files in data/raw/\n"
        "python run_pipeline.py\n"
        "```"
    )


# --------------------------------------------------------------------------
# Shared input helper
# --------------------------------------------------------------------------
def build_input_row(master: pd.DataFrame, key_prefix: str) -> tuple[pd.DataFrame, dict]:
    """
    Collect visitor and attraction details, and assemble a single master-schema
    row the models can score.

    Geography selectors cascade (continent filters country filters city) so the
    user cannot construct a combination that never occurs in the data.
    """
    left, right = st.columns(2)

    with left:
        st.markdown("**Where the visitor is from**")
        continent = st.selectbox(
            "Continent", sorted(master.Continent.dropna().unique()), key=f"{key_prefix}_cont"
        )
        sub = master[master.Continent == continent]
        country = st.selectbox(
            "Country", sorted(sub.Country.dropna().unique()), key=f"{key_prefix}_country"
        )
        sub2 = sub[sub.Country == country]
        city = st.selectbox(
            "Home city", sorted(sub2.CityName.dropna().unique()), key=f"{key_prefix}_city"
        )
        region = sub2.Region.mode().iloc[0] if len(sub2) else "Unknown"

    with right:
        st.markdown("**The visit**")
        a_type = st.selectbox(
            "Attraction type", sorted(master.AttractionType.dropna().unique()),
            key=f"{key_prefix}_type",
        )
        pool = master[master.AttractionType == a_type]
        attraction = st.selectbox(
            "Attraction", sorted(pool.Attraction.dropna().unique()), key=f"{key_prefix}_attr"
        )
        month = st.select_slider(
            "Month of visit", options=list(range(1, 13)), value=7, key=f"{key_prefix}_month"
        )
        year = st.select_slider(
            "Year",
            options=sorted(master.VisitYear.unique()),
            value=int(master.VisitYear.max()),
            key=f"{key_prefix}_year",
        )

    match = pool[pool.Attraction == attraction]
    row = {
        "UserId": -1,  # unknown visitor -> aggregates fall back to global priors
        "AttractionId": int(match.AttractionId.iloc[0]) if len(match) else -1,
        "Continent": continent,
        "Region": region,
        "Country": country,
        "CityName": city,
        "AttractionType": a_type,
        "Attraction": attraction,
        "AttractionCity": match.AttractionCity.iloc[0] if len(match) else "Unknown",
        "VisitYear": int(year),
        "VisitMonth": int(month),
    }
    return pd.DataFrame([row]), row


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(master, metrics):
    st.title("Tourism Experience Analytics")
    st.markdown(
        '<p class="lede">Predicting visitor satisfaction, classifying how people travel, '
        'and recommending attractions — built on visit history across regions, '
        'attraction types and travel parties.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    stats = [
        (f"{len(master):,}", "Visits analysed"),
        (f"{master.UserId.nunique():,}", "Visitors"),
        (f"{master.AttractionId.nunique():,}", "Attractions"),
        (f"{master.Rating.mean():.2f}", "Mean rating"),
        (f"{master.Country.nunique():,}", "Source countries"),
    ]
    for col, (v, l) in zip(cols, stats):
        col.markdown(metric_card(v, l), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Model scorecard")

    reg, clf, rec = metrics.get("regression"), metrics.get("classification"), metrics.get("recommendation")
    c1, c2, c3 = st.columns(3)

    if reg:
        c1.markdown(
            metric_card(f"{reg['test_r2']:.3f}", f"Rating R² — {reg['best_model']}"),
            unsafe_allow_html=True,
        )
        c1.caption(
            f"RMSE {reg['test_rmse']:.3f} · 5-fold CV R² {reg['cv_r2_mean']:.3f} "
            f"± {reg['cv_r2_std']:.3f}"
        )
    if clf:
        weak = clf["test_f1_macro"] < 0.40
        c2.markdown(
            metric_card(f"{clf['test_f1_macro']:.3f}", f"Visit mode macro F1 — {clf['best_model']}",
                        attention=weak),
            unsafe_allow_html=True,
        )
        base = 1 / max(len(clf.get("classes", [1])), 1)
        c2.caption(f"Accuracy {clf['test_accuracy']:.3f} · random-guess F1 ≈ {base:.3f}")
    if rec:
        best = rec["comparison"][0] if rec["comparison"] else {}
        c3.markdown(
            metric_card(f"{best.get('ndcg@10', float('nan')):.3f}",
                        f"Recommendation NDCG@10 — {rec['best_method']}"),
            unsafe_allow_html=True,
        )
        c3.caption(f"Hit rate {best.get('hit_rate@10', 0):.3f} · MAP {best.get('map@10', 0):.3f}")

    if clf and clf["test_f1_macro"] < 0.40:
        st.markdown(
            '<div class="note"><b>Read the visit-mode model with care.</b> It beats the '
            'majority-class baseline by a wide margin, but its absolute accuracy is low: '
            'visitor origin and attraction choice only weakly determine who someone travels '
            'with. Use its probabilities to rank campaign audiences, not to make a call about '
            'any single visitor.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Where the demand is")
    left, right = st.columns([3, 2])
    with left:
        by_country = (
            master.groupby("Country").agg(visits=("Rating", "size"), rating=("Rating", "mean"))
            .sort_values("visits", ascending=False).head(15)
        )
        st.bar_chart(
            by_country["visits"].sort_values(), color="#0E7C7B", height=430, horizontal=True
        )
    with right:
        st.dataframe(
            by_country.assign(rating=by_country.rating.round(2))
            .rename(columns={"visits": "Visits", "rating": "Mean rating"}),
            height=430, width="stretch",
        )


def page_explore(master, metrics):
    st.title("Explore the data")
    st.markdown(
        '<p class="lede">The exploratory analysis and the SQL layer behind the '
        'headline numbers.</p>',
        unsafe_allow_html=True,
    )

    tab_find, tab_fig, tab_sql, tab_clean = st.tabs(
        ["Findings", "Figures", "SQL analysis", "Cleaning log"]
    )

    with tab_find:
        findings = metrics.get("eda", {}).get("findings", [])
        if findings:
            for f in findings:
                st.markdown(f"- {f}")
        else:
            st.info("Run `python run_pipeline.py` to generate the analysis.")

    with tab_fig:
        figures = sorted(FIGURE_DIR.glob("*.png"))
        if not figures:
            st.info("No figures yet. Run the pipeline to generate them.")
        for fig in figures:
            title = fig.stem.split("_", 1)[-1].replace("_", " ").capitalize()
            st.markdown(f"**{title}**")
            st.image(str(fig), width="stretch")

    with tab_sql:
        if not SQLITE_DB.exists():
            st.info("No database yet. Run the pipeline to build it.")
        else:
            from src.database import parse_queries
            queries = parse_queries()
            names = list(queries)
            choice = st.selectbox(
                "Query", names,
                format_func=lambda n: n.replace("_", " ").capitalize(),
            )
            st.code(queries[choice], language="sql")
            try:
                st.dataframe(run_sql(choice), width="stretch", height=400)
            except Exception as exc:
                st.error(f"Query failed: {exc}")

    with tab_clean:
        if CLEANING_LOG.exists():
            st.markdown(CLEANING_LOG.read_text(encoding="utf-8"))
        else:
            st.info("No cleaning log yet.")


def page_rating(master, metrics):
    st.title("Predict a rating")
    st.markdown(
        '<p class="lede">Estimate the rating a visitor is likely to give, so low-scoring '
        'matches can be caught before the trip rather than after the review.</p>',
        unsafe_allow_html=True,
    )

    artefact = load_regressor()
    if artefact is None:
        st.warning("No regression model found. Run `python run_pipeline.py` first.")
        return

    frame, row = build_input_row(master, "reg")

    if st.button("Predict rating", type="primary"):
        from src.models import regression
        pred = float(regression.predict(artefact, frame)[0])

        peers = master[master.Attraction == row["Attraction"]]
        actual = peers.Rating.mean() if len(peers) else float("nan")
        overall = master.Rating.mean()

        c1, c2, c3 = st.columns(3)
        c1.markdown(
            metric_card(f"{pred:.2f}", "Predicted rating", attention=pred < overall - 0.25),
            unsafe_allow_html=True,
        )
        c2.markdown(metric_card(f"{actual:.2f}", "This attraction's historical mean"),
                    unsafe_allow_html=True)
        c3.markdown(metric_card(f"{overall:.2f}", "Overall mean"), unsafe_allow_html=True)

        if pred < overall - 0.25:
            st.markdown(
                '<div class="note"><b>Below-average match.</b> Consider setting expectations '
                'more carefully in the listing, or steering this visitor toward a better-fitting '
                'attraction — see the Recommend page.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="note">This pairing scores at or above the overall average. '
                'Safe to promote to this audience.</div>',
                unsafe_allow_html=True,
            )

        st.caption(
            f"Model: {artefact['best_model_name']}. Because the visitor is new, "
            "user-level history falls back to population averages — the prediction reflects "
            "the attraction and trip context rather than personal taste."
        )

    reg = metrics.get("regression")
    if reg:
        with st.expander("How this model was chosen"):
            st.dataframe(pd.DataFrame(reg["comparison"]), width="stretch")
            st.markdown("**Most influential features** (permutation importance on held-out data)")
            st.dataframe(pd.DataFrame(reg["top_features"]).head(10), width="stretch")


def page_visit_mode(master, metrics):
    st.title("Predict visit mode")
    st.markdown(
        '<p class="lede">Classify how a visitor is travelling — alone, as a couple, with '
        'family or friends, or on business — to target packages and plan on-site resources.</p>',
        unsafe_allow_html=True,
    )

    artefact = load_classifier()
    if artefact is None:
        st.warning("No classification model found. Run `python run_pipeline.py` first.")
        return

    frame, row = build_input_row(master, "clf")
    rating = st.slider(
        "Rating given (if already known)", 1.0, 5.0, float(round(master.Rating.mean(), 1)), 0.5,
        help="Visit mode is usually inferred after the fact, when the rating is known.",
    )
    frame["Rating"] = rating

    if st.button("Predict visit mode", type="primary"):
        from src.models import classification
        label, proba = classification.predict(artefact, frame, return_proba=True)
        probs = proba.iloc[0].sort_values(ascending=False)

        top, second = probs.index[0], probs.index[1]
        margin = probs.iloc[0] - probs.iloc[1]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(
                metric_card(str(label[0]), f"Most likely mode · {probs.iloc[0]:.0%} confidence",
                            attention=margin < 0.10),
                unsafe_allow_html=True,
            )
        with c2:
            st.bar_chart(probs, color="#0E7C7B", height=240)

        if margin < 0.10:
            st.markdown(
                f'<div class="note"><b>Close call.</b> {top} and {second} are separated by only '
                f'{margin:.0%}. Treat this visitor as belonging to either segment rather than '
                'committing to one campaign.</div>',
                unsafe_allow_html=True,
            )

    clf = metrics.get("classification")
    if clf:
        with st.expander("Per-class performance"):
            st.dataframe(pd.DataFrame(clf["per_class"]).T, width="stretch")
            st.caption(
                "Precision and recall are reported per class because the modes are "
                "imbalanced — overall accuracy hides how the rare segments perform."
            )
            st.markdown("**Model comparison**")
            st.dataframe(pd.DataFrame(clf["comparison"]), width="stretch")


def page_recommend(master, metrics):
    st.title("Recommend attractions")
    st.markdown(
        '<p class="lede">Rank attractions a visitor has not seen yet, blending what similar '
        'visitors enjoyed with how closely an attraction matches their own history.</p>',
        unsafe_allow_html=True,
    )

    model = load_recommender()
    if model is None:
        st.warning("No recommender found. Run `python run_pipeline.py` first.")
        return

    mode = st.radio(
        "Who are you recommending for?",
        ["An existing visitor", "A new visitor"],
        horizontal=True,
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        if mode == "An existing visitor":
            known = list(model.user_ids_)
            user_id = st.selectbox(
                "Visitor", known[:3000],
                format_func=lambda u: f"Visitor {u}",
            )
        else:
            user_id = -1
            st.markdown(
                '<div class="note">New visitors have no history, so the system ranks by '
                'overall popularity and satisfaction. It switches to personalised results as '
                'soon as they rate their first attraction.</div>',
                unsafe_allow_html=True,
            )
    with c2:
        method = st.selectbox(
            "Method", ["hybrid", "collaborative", "content", "svd", "popularity"],
            help="Hybrid scored best on held-out data.",
        )
        n = st.slider("How many", 5, 20, TOP_N_RECOMMENDATIONS)

    if mode == "An existing visitor":
        history = master[master.UserId == user_id]
        if len(history):
            st.markdown(f"**Visit history** — {len(history)} visits, "
                        f"mean rating {history.Rating.mean():.2f}")
            st.dataframe(
                history[["Attraction", "AttractionType", "AttractionCity",
                         "VisitMode", "VisitYear", "Rating"]]
                .sort_values("VisitYear", ascending=False).head(10),
                width="stretch", hide_index=True,
            )

    if st.button("Get recommendations", type="primary"):
        recs = model.recommend(user_id, n=n, method=method)
        if recs.empty:
            st.info("No recommendations available for this visitor.")
            return

        cols = [c for c in ["Attraction", "AttractionType", "AttractionCity",
                            "AvgRating", "VisitCount", "Score"] if c in recs.columns]
        st.dataframe(
            recs[cols].round({"Score": 3}),
            width="stretch", hide_index=True,
        )
        st.caption(f"Ranked by: {recs['Method'].iloc[0]}")

        with st.expander("More like a specific attraction"):
            pick = st.selectbox("Attraction", recs.Attraction.tolist())
            aid = recs.loc[recs.Attraction == pick, "AttractionId"].iloc[0]
            similar = model.similar_attractions(aid, n=8)
            if not similar.empty:
                st.dataframe(similar, width="stretch", hide_index=True)

    rec = metrics.get("recommendation")
    if rec and rec.get("comparison"):
        with st.expander("How the methods compare on held-out visits"):
            st.dataframe(pd.DataFrame(rec["comparison"]), width="stretch")
            st.caption(
                "Evaluated by holding out each visitor's most recent visit and asking the "
                "model to rank everything they had not already seen. A random split would "
                "let the model use future visits to predict past ones."
            )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
def main():
    master = load_master()
    metrics = load_metrics()

    st.sidebar.markdown("### Tourism Analytics")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Explore the data", "Predict a rating", "Predict visit mode",
         "Recommend attractions"],
        label_visibility="collapsed",
    )

    if master is None:
        pipeline_missing()
        return

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"{len(master):,} visits · {master.UserId.nunique():,} visitors · "
        f"{master.AttractionId.nunique():,} attractions"
    )
    if metrics.get("generated_at"):
        st.sidebar.caption(f"Pipeline run: {metrics['generated_at']}")

    {
        "Overview": page_overview,
        "Explore the data": page_explore,
        "Predict a rating": page_rating,
        "Predict visit mode": page_visit_mode,
        "Recommend attractions": page_recommend,
    }[page](master, metrics)


if __name__ == "__main__":
    main()
