"""
Task 3 -- Recommendation: personalised attraction suggestions.

Three approaches are implemented and compared, as the brief asks:

  * **Collaborative filtering** -- item-item cosine similarity over the
    mean-centred user-item rating matrix, plus a truncated-SVD latent factor
    model. Answers "people who liked what you liked also liked...".

  * **Content-based filtering** -- cosine similarity over attraction
    attributes (type, city, country, popularity band). Answers "more things
    like the ones you already enjoyed", and is the fallback for cold-start
    users, where collaborative filtering has nothing to work with.

  * **Hybrid** -- a weighted blend of the two, which is the default served by
    the app.

**Evaluation.** Ranking quality is measured with a temporal leave-last-out
protocol: for every user with enough history, their most recent visit is held
out and the model is asked to rank all attractions they have not already seen.
A held-out item counts as relevant if the user rated it 4 or higher. Reported
metrics are Hit Rate@K, Precision@K, Recall@K, MAP@K and NDCG@K, alongside
RMSE for the SVD model's rating reconstruction.

Evaluating on a random split instead of a temporal one would let the model use
a user's future visits to predict their past ones, which flatters the score and
is not how the system is used in production.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from ..config import (
    MIN_ITEM_INTERACTIONS,
    MIN_USER_INTERACTIONS,
    RANDOM_STATE,
    RECO_MODEL_PATH,
    TOP_N_RECOMMENDATIONS,
)

RELEVANCE_THRESHOLD = 4  # rating at or above which a held-out item is "relevant"


# --------------------------------------------------------------------------
# Ranking metrics
# --------------------------------------------------------------------------
def _dcg(relevances: np.ndarray) -> float:
    return float(np.sum(relevances / np.log2(np.arange(2, relevances.size + 2))))


def ranking_metrics(recommended: list, relevant: set, k: int) -> dict[str, float]:
    """Precision@K, Recall@K, Average Precision@K, NDCG@K and hit rate."""
    rec_k = recommended[:k]
    if not relevant:
        return {}

    hits = np.array([1.0 if item in relevant else 0.0 for item in rec_k])
    n_hits = hits.sum()

    precision = n_hits / k
    recall = n_hits / len(relevant)

    # Average precision: mean of precision@i at each hit position.
    if n_hits:
        cum = np.cumsum(hits)
        positions = np.arange(1, len(rec_k) + 1)
        ap = float(np.sum((cum / positions) * hits) / min(len(relevant), k))
    else:
        ap = 0.0

    ideal = np.ones(min(len(relevant), k))
    ndcg = _dcg(hits) / _dcg(ideal) if ideal.size else 0.0

    return {
        "hit_rate": float(n_hits > 0),
        "precision": float(precision),
        "recall": float(recall),
        "ap": ap,
        "ndcg": float(ndcg),
    }


# --------------------------------------------------------------------------
# Recommender
# --------------------------------------------------------------------------
class TourismRecommender:
    """Collaborative + content-based + hybrid recommender over the master table."""

    def __init__(
        self,
        n_factors: int = 64,
        w_collaborative: float = 0.45,
        w_content: float = 0.20,
        w_popularity: float = 0.35,
        cf_shrinkage: float = 2.0,
    ):
        """
        The hybrid blends three signals. The popularity term is not a
        concession -- a personalised score alone ignores that some attractions
        are simply far more visited, and a production recommender that ignores
        the prior performs worse than one that does not. Weights are exposed so
        they can be retuned per dataset.
        """
        self.n_factors = n_factors
        self.w_collaborative = w_collaborative
        self.w_content = w_content
        self.w_popularity = w_popularity
        self.cf_shrinkage = cf_shrinkage
        self.fitted_ = False

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame) -> "TourismRecommender":
        """Build the user-item matrix, item similarities and content profiles."""
        # Deduplicate repeat visits to the same attraction by averaging.
        ratings = (
            df.groupby(["UserId", "AttractionId"], observed=True)["Rating"]
            .mean()
            .reset_index()
        )

        # Trim users/items with too little history to support a similarity.
        u_counts = ratings.UserId.value_counts()
        i_counts = ratings.AttractionId.value_counts()
        ratings = ratings[
            ratings.UserId.isin(u_counts[u_counts >= MIN_USER_INTERACTIONS].index)
            & ratings.AttractionId.isin(i_counts[i_counts >= MIN_ITEM_INTERACTIONS].index)
        ]

        self.user_ids_ = np.sort(ratings.UserId.unique())
        self.item_ids_ = np.sort(ratings.AttractionId.unique())
        self.user_index_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_index_ = {a: i for i, a in enumerate(self.item_ids_)}

        rows = ratings.UserId.map(self.user_index_).to_numpy()
        cols = ratings.AttractionId.map(self.item_index_).to_numpy()
        vals = ratings.Rating.to_numpy(dtype=float)

        shape = (len(self.user_ids_), len(self.item_ids_))
        self.R_ = sparse.csr_matrix((vals, (rows, cols)), shape=shape)
        self.seen_ = sparse.csr_matrix((np.ones_like(vals), (rows, cols)), shape=shape)

        self.global_mean_ = float(vals.mean())
        counts = np.asarray(self.seen_.sum(axis=1)).ravel()
        sums = np.asarray(self.R_.sum(axis=1)).ravel()
        self.user_mean_ = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_mean_)

        # --- collaborative: item-item similarity on mean-centred ratings ----
        # Centring removes each user's personal generosity, so similarity
        # reflects agreement on *relative* preference, not shared optimism.
        R_centered = self.R_.copy().astype(float)
        R_centered.data = R_centered.data - self.user_mean_[
            np.repeat(np.arange(shape[0]), np.diff(self.R_.indptr))
        ]
        self.R_centered_ = R_centered
        self.item_sim_ = cosine_similarity(R_centered.T.tocsr(), dense_output=True)
        np.fill_diagonal(self.item_sim_, 0.0)

        # --- collaborative: latent factors ---------------------------------
        n_comp = int(min(self.n_factors, min(shape) - 1))
        if n_comp >= 2:
            self.svd_ = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
            self.user_factors_ = self.svd_.fit_transform(R_centered)
            self.item_factors_ = self.svd_.components_.T
        else:  # pragma: no cover - only on degenerate data
            self.svd_ = None
            self.user_factors_ = np.zeros((shape[0], 1))
            self.item_factors_ = np.zeros((shape[1], 1))

        # --- content-based profiles ----------------------------------------
        self._build_content_profiles(df)

        # Item popularity, used for cold-start and as a tie-break.
        pop = ratings.groupby("AttractionId").size()
        self.item_popularity_ = np.array(
            [pop.get(a, 0) for a in self.item_ids_], dtype=float
        )
        avg = ratings.groupby("AttractionId")["Rating"].mean()
        self.item_avg_rating_ = np.array(
            [avg.get(a, self.global_mean_) for a in self.item_ids_], dtype=float
        )

        self.fitted_ = True
        return self

    def _build_content_profiles(self, df: pd.DataFrame) -> None:
        """One-hot attraction attributes -> cosine similarity between items."""
        meta = (
            df.drop_duplicates("AttractionId")
            .set_index("AttractionId")
            .reindex(self.item_ids_)
        )
        self.item_meta_ = meta[
            [c for c in ["Attraction", "AttractionType", "AttractionCity", "AttractionAddress"]
             if c in meta.columns]
        ].copy()

        feature_cols = [c for c in ["AttractionType", "AttractionCity"] if c in meta.columns]
        dummies = pd.get_dummies(meta[feature_cols].astype(str), dummy_na=False)

        # Attraction type carries more signal than city for "similar to what I
        # liked", so it is weighted up before normalising.
        for col in dummies.columns:
            if col.startswith("AttractionType"):
                dummies[col] = dummies[col].astype(float) * 2.0

        M = normalize(dummies.to_numpy(dtype=float))
        self.content_sim_ = cosine_similarity(M, dense_output=True)
        np.fill_diagonal(self.content_sim_, 0.0)

    # ------------------------------------------------------------- scoring
    def _user_vector(self, user_id) -> tuple[np.ndarray, np.ndarray] | None:
        idx = self.user_index_.get(user_id)
        if idx is None:
            return None
        row = self.R_.getrow(idx)
        return row.indices, row.data

    def _collaborative_scores(self, user_id) -> np.ndarray:
        """
        Item-item CF score: similarity-weighted average of the user's ratings.

        The raw weighted average is shrunk by how much similarity actually
        supports it. Without this, an item that happens to share one weak
        neighbour with the user's history can score as highly as one backed by
        the user's whole profile, and thinly-rated items dominate the top-N.
        """
        uv = self._user_vector(user_id)
        if uv is None:
            return np.zeros(len(self.item_ids_))
        item_idx, ratings = uv
        centred = ratings - self.user_mean_[self.user_index_[user_id]]

        sims = self.item_sim_[item_idx]                 # (n_rated, n_items)
        support = np.abs(sims).sum(axis=0)
        raw = (centred @ sims) / (support + 1e-9)

        # Empirical-Bayes style shrinkage toward 0 (= "no opinion").
        confidence = support / (support + self.cf_shrinkage)
        return raw * confidence

    def _svd_scores(self, user_id) -> np.ndarray:
        idx = self.user_index_.get(user_id)
        if idx is None or self.svd_ is None:
            return np.zeros(len(self.item_ids_))
        return self.user_factors_[idx] @ self.item_factors_.T

    def _content_scores(self, user_id) -> np.ndarray:
        """Average content similarity to the items this user rated well."""
        uv = self._user_vector(user_id)
        if uv is None:
            return np.zeros(len(self.item_ids_))
        item_idx, ratings = uv
        liked = item_idx[ratings >= RELEVANCE_THRESHOLD]
        if liked.size == 0:
            liked = item_idx  # fall back to everything they visited
        return self.content_sim_[liked].mean(axis=0)

    @staticmethod
    def _standardise(x: np.ndarray) -> np.ndarray:
        s = x.std()
        return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

    def score(self, user_id, method: str = "hybrid") -> np.ndarray:
        """Score every item for a user under the requested method."""
        if method == "collaborative":
            return self._collaborative_scores(user_id)
        if method == "svd":
            return self._svd_scores(user_id)
        if method == "content":
            return self._content_scores(user_id)
        if method == "popularity":
            # log-damped visit count, so a runaway attraction does not swamp
            # everything, combined with average rating.
            return self._standardise(np.log1p(self.item_popularity_)) + self._standardise(
                self.item_avg_rating_
            )
        if method == "hybrid":
            return (
                self.w_collaborative * self._standardise(self._collaborative_scores(user_id))
                + self.w_content * self._standardise(self._content_scores(user_id))
                + self.w_popularity * self._standardise(self.score(user_id, "popularity"))
            )
        raise ValueError(f"Unknown method: {method!r}")

    # ------------------------------------------------------- public API
    def recommend(
        self,
        user_id,
        n: int = TOP_N_RECOMMENDATIONS,
        method: str = "hybrid",
        exclude_seen: bool = True,
    ) -> pd.DataFrame:
        """
        Top-N attractions for a user, as a readable table.

        Unknown users fall back to a popularity ranking -- the honest cold-start
        answer, rather than an arbitrary one dressed up as personalisation.
        """
        if not self.fitted_:
            raise RuntimeError("Recommender not fitted.")

        known = user_id in self.user_index_
        if not known:
            method = "popularity"
        scores = self.score(user_id, method=method).astype(float)

        if exclude_seen and known:
            seen_idx = self.R_.getrow(self.user_index_[user_id]).indices
            scores[seen_idx] = -np.inf

        top = np.argsort(-scores)[:n]
        out = pd.DataFrame(
            {
                "AttractionId": self.item_ids_[top],
                "Score": scores[top],
                "AvgRating": self.item_avg_rating_[top].round(2),
                "VisitCount": self.item_popularity_[top].astype(int),
            }
        )
        meta = self.item_meta_.reindex(out.AttractionId)
        for c in ("Attraction", "AttractionType", "AttractionCity"):
            if c in meta.columns:
                out[c] = meta[c].to_numpy()
        out["Method"] = "popularity (cold start)" if not known else method
        return out.reset_index(drop=True)

    def similar_attractions(self, attraction_id, n: int = 10) -> pd.DataFrame:
        """Content-similar attractions -- the 'more like this' panel."""
        idx = self.item_index_.get(attraction_id)
        if idx is None:
            return pd.DataFrame()
        sims = self.content_sim_[idx].copy()
        top = np.argsort(-sims)[:n]
        out = pd.DataFrame(
            {"AttractionId": self.item_ids_[top], "Similarity": sims[top].round(4)}
        )
        meta = self.item_meta_.reindex(out.AttractionId)
        for c in ("Attraction", "AttractionType", "AttractionCity"):
            if c in meta.columns:
                out[c] = meta[c].to_numpy()
        return out


# --------------------------------------------------------------------------
# Temporal evaluation
# --------------------------------------------------------------------------
def temporal_split(df: pd.DataFrame, min_history: int = 3):
    """
    Hold out each qualifying user's single most recent interaction.

    Ordering is by (VisitYear, VisitMonth, TransactionId); the transaction id
    breaks ties within a month so the split is deterministic.
    """
    d = df.copy()
    sort_cols = [c for c in ["VisitYear", "VisitMonth", "TransactionId"] if c in d.columns]
    d = d.sort_values(["UserId"] + sort_cols)

    counts = d.groupby("UserId")["AttractionId"].transform("size")
    eligible = counts >= min_history

    is_last = d.groupby("UserId").cumcount(ascending=False) == 0
    holdout_mask = eligible & is_last

    return d[~holdout_mask].copy(), d[holdout_mask].copy()


def evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: TourismRecommender,
    k: int = TOP_N_RECOMMENDATIONS,
    methods: tuple[str, ...] = ("popularity", "content", "collaborative", "svd", "hybrid"),
    max_users: int = 2000,
) -> pd.DataFrame:
    """Compare every method on the held-out interactions."""
    rng = np.random.default_rng(RANDOM_STATE)

    # Only users the model knows and whose held-out item is in the catalogue.
    test = test_df[
        test_df.UserId.isin(model.user_index_)
        & test_df.AttractionId.isin(model.item_index_)
        & (test_df.Rating >= RELEVANCE_THRESHOLD)
    ]
    users = test.UserId.unique()
    if len(users) > max_users:
        users = rng.choice(users, size=max_users, replace=False)
        test = test[test.UserId.isin(users)]

    truth = test.groupby("UserId")["AttractionId"].apply(set).to_dict()

    rows = []
    for method in methods:
        acc = []
        for user_id, relevant in truth.items():
            recs = model.recommend(user_id, n=k, method=method)
            m = ranking_metrics(recs.AttractionId.tolist(), relevant, k)
            if m:
                acc.append(m)
        if not acc:
            continue
        agg = pd.DataFrame(acc).mean()
        rows.append(
            {
                "method": method,
                f"hit_rate@{k}": round(float(agg["hit_rate"]), 4),
                f"precision@{k}": round(float(agg["precision"]), 4),
                f"recall@{k}": round(float(agg["recall"]), 4),
                f"map@{k}": round(float(agg["ap"]), 4),
                f"ndcg@{k}": round(float(agg["ndcg"]), 4),
                "users_evaluated": len(acc),
            }
        )

    return pd.DataFrame(rows).sort_values(f"ndcg@{k}", ascending=False).reset_index(drop=True)


def rating_rmse(model: TourismRecommender, test_df: pd.DataFrame) -> float:
    """
    RMSE of the SVD model's reconstructed ratings on held-out interactions.

    Reported because the brief names RMSE as a recommender metric, though for a
    top-N system the ranking metrics above matter far more: a system is judged
    on what it puts in front of the user, not on how well it reconstructs
    ratings for items the user will never be shown.
    """
    preds, actuals = [], []
    for row in test_df.itertuples():
        ui = model.user_index_.get(row.UserId)
        ii = model.item_index_.get(row.AttractionId)
        if ui is None or ii is None:
            continue
        est = model.user_mean_[ui] + float(model.user_factors_[ui] @ model.item_factors_[ii])
        preds.append(np.clip(est, 1, 5))
        actuals.append(float(row.Rating))
    if not preds:
        return float("nan")
    return float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2)))


def train(master: pd.DataFrame, verbose: bool = True) -> dict:
    """Fit, evaluate and persist the recommender."""
    train_df, test_df = temporal_split(master)

    eval_model = TourismRecommender().fit(train_df)
    comparison = evaluate(train_df, test_df, eval_model)
    rmse = rating_rmse(eval_model, test_df)

    if verbose:
        print(comparison.to_string(index=False))
        print(f"    SVD rating RMSE on held-out interactions: {rmse:.4f}")

    # Refit on the full dataset for serving -- the split existed only to
    # produce an honest estimate.
    final = TourismRecommender().fit(master)
    joblib.dump(final, RECO_MODEL_PATH, compress=3)

    best = comparison.iloc[0]["method"] if len(comparison) else "hybrid"
    if verbose:
        print(f"    -> best ranking method: {best}")

    return {
        "task": "recommendation",
        "best_method": best,
        "k": TOP_N_RECOMMENDATIONS,
        "svd_rating_rmse": round(rmse, 4),
        "comparison": comparison.to_dict(orient="records"),
        "n_users": int(len(final.user_ids_)),
        "n_items": int(len(final.item_ids_)),
        "n_train_interactions": int(len(train_df)),
        "n_holdout_users": int(comparison.iloc[0]["users_evaluated"]) if len(comparison) else 0,
        "model_path": str(RECO_MODEL_PATH),
    }


def load_model(path=RECO_MODEL_PATH) -> TourismRecommender:
    return joblib.load(path)
