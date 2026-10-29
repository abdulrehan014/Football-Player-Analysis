"""Streamlit UI for the Football Player Data Analysis notebook.

Run:  streamlit run app.py
Data: put players.csv next to this file, or upload it from the sidebar.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from statsmodels.formula.api import ols

st.set_page_config(page_title="Football Player Analytics", page_icon="⚽", layout="wide")

DEFAULT_CSV = Path(__file__).parent / "players.csv"

MONEY_COLS = ["Value", "Wage", "Release clause"]
RATING_COLS = [
    "Overall rating", "Potential", "Crossing", "Finishing", "Heading accuracy",
    "Short passing", "Volleys", "Dribbling", "Curve", "FK Accuracy", "Long passing",
    "Ball control", "Acceleration", "Sprint speed", "Agility", "Reactions", "Balance",
    "Shot power", "Jumping", "Stamina", "Strength", "Long shots", "Aggression",
    "Interceptions", "Att. Position", "Vision", "Penalties", "Composure",
    "Defensive awareness", "Standing tackle", "Sliding tackle", "GK Diving",
    "GK Handling", "GK Kicking", "GK Positioning", "GK Reflexes",
]


# ----------------------------------------------------------------- data ----
def parse_money(x):
    """'€1.5M' -> 1_500_000, '€900K' -> 900_000, '€0' -> 0. Safe (no eval)."""
    s = str(x).replace("€", "").strip()
    m = re.fullmatch(r"([\d.]+)\s*([KkMm]?)", s)
    if not m:
        return np.nan
    mult = {"": 1, "k": 1e3, "m": 1e6}[m.group(2).lower()]
    return float(m.group(1)) * mult


def parse_rating(x):
    """'87+1' -> 87. Values above 100 become NaN."""
    s = re.sub(r"[^\d.]", "", str(x).split("+")[0].strip())
    v = pd.to_numeric(s, errors="coerce")
    return np.nan if pd.notna(v) and v > 100 else v


@st.cache_data(show_spinner="Cleaning data…")
def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    for c in MONEY_COLS:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(parse_money)
    for c in RATING_COLS:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].map(parse_rating).astype(float)
    if "Team & Contract" in df:
        parts = df["Team & Contract"].astype(str).str.split("\n", n=1, expand=True)
        df["Team"] = parts[0].str.strip()
    for c in ("Height", "Weight"):  # cm / kg are easy to pull out of "182cm / 6'0\""
        if c in df:
            unit = "cm" if c == "Height" else "kg"
            df[f"{c} ({unit})"] = pd.to_numeric(
                df[c].astype(str).str.extract(rf"(\d+)\s*{unit}")[0], errors="coerce"
            )
            df = df.drop(columns=c)
    return df


@st.cache_data
def load_csv(source) -> pd.DataFrame:
    return pd.read_csv(source)


# -------------------------------------------------------------- sidebar ----
st.sidebar.title("⚽ Football Analytics")
upload = st.sidebar.file_uploader("Upload players.csv", type="csv")
if upload is not None:
    raw = load_csv(upload)
elif DEFAULT_CSV.exists():
    raw = load_csv(DEFAULT_CSV)
else:
    st.title("⚽ Football Player Data Analysis")
    st.info("Upload **players.csv** in the sidebar (or place it next to `app.py`) to begin.")
    st.stop()

full = clean(raw)

st.sidebar.markdown("### Filters")
age_lo, age_hi = int(full["Age"].min()), int(full["Age"].max())
age_rng = st.sidebar.slider("Age", age_lo, age_hi, (age_lo, age_hi))
ovr = full["Overall rating"].dropna()
ovr_rng = st.sidebar.slider("Overall rating", int(ovr.min()), int(ovr.max()),
                            (int(ovr.min()), int(ovr.max())))
positions = sorted(full["Best position"].dropna().unique())
pos_sel = st.sidebar.multiselect("Best position", positions)
feet = sorted(full["foot"].dropna().unique())
foot_sel = st.sidebar.multiselect("Foot", feet)

df = full[full["Age"].between(*age_rng)]
df = df[df["Overall rating"].between(*ovr_rng)]
if pos_sel:
    df = df[df["Best position"].isin(pos_sel)]
if foot_sel:
    df = df[df["foot"].isin(foot_sel)]
st.sidebar.caption(f"{len(df):,} of {len(full):,} players selected")

if df.empty:
    st.warning("No players match the filters.")
    st.stop()

num_cols = df.select_dtypes("number").columns.tolist()
st.title("⚽ Football Player Data Analysis")

tabs = st.tabs([
    "Overview", "Descriptive stats", "Visualizations", "Sampling & CLT",
    "Hypothesis tests", "Correlation & ANOVA", "Value prediction", "Player explorer",
])

# ------------------------------------------------------------- overview ----
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Players", f"{len(df):,}")
    c[1].metric("Avg overall", f"{df['Overall rating'].mean():.1f}")
    c[2].metric("Avg potential", f"{df['Potential'].mean():.1f}")
    c[3].metric("Avg age", f"{df['Age'].mean():.1f}")
    c[4].metric("Median value", f"€{df['Value'].median():,.0f}")
    st.subheader("Data preview")
    st.dataframe(df.head(100), width='stretch')
    left, right = st.columns(2)
    with left:
        st.subheader("Missing values (after cleaning)")
        miss = df.isna().sum()
        miss = miss[miss > 0].rename("missing").to_frame()
        st.dataframe(miss if len(miss) else pd.DataFrame({"missing": ["none"]}))
    with right:
        st.subheader("Column types")
        st.dataframe(df.dtypes.astype(str).rename("dtype").to_frame(), height=250)
    st.download_button("Download cleaned data (CSV)", df.to_csv(index=False).encode(),
                       "players_clean.csv", "text/csv")

# ---------------------------------------------------------- descriptive ----
with tabs[1]:
    st.subheader("Summary statistics")
    st.dataframe(df[num_cols].describe().T, width='stretch')
    st.subheader("Detailed measures")
    cols = st.multiselect("Columns", num_cols,
                          default=[c for c in ["Age", "Overall rating", "Potential", "Value", "Wage"] if c in num_cols])
    if cols:
        rows = {}
        for c in cols:
            s = df[c].dropna()
            mode = s.mode()
            rows[c] = {
                "Mean": s.mean(), "Median": s.median(),
                "Mode": mode.iloc[0] if len(mode) else np.nan,
                "Variance": s.var(), "Std dev": s.std(),
                "Q1": s.quantile(.25), "Q2": s.quantile(.5), "Q3": s.quantile(.75),
                "IQR": s.quantile(.75) - s.quantile(.25),
                "Skewness": s.skew(), "Kurtosis": s.kurt(),
            }
        st.dataframe(pd.DataFrame(rows).T.style.format("{:,.2f}"), width='stretch')

# -------------------------------------------------------- visualizations ----
with tabs[2]:
    kind = st.radio("Chart", ["Histogram", "Box plot", "Pie / bar (categorical)",
                              "Scatter", "Average by age"], horizontal=True)
    if kind == "Histogram":
        c = st.selectbox("Column", num_cols, index=num_cols.index("Overall rating"))
        bins = st.slider("Bins", 5, 100, 30)
        st.plotly_chart(px.histogram(df, x=c, nbins=bins, marginal="violin"),
                        width='stretch')
    elif kind == "Box plot":
        c = st.selectbox("Column", num_cols, index=num_cols.index("Value") if "Value" in num_cols else 0)
        by = st.selectbox("Group by", [None, "foot", "Best position", "International reputation"])
        st.plotly_chart(px.box(df, y=c, x=by, points="outliers"), width='stretch')
    elif kind == "Pie / bar (categorical)":
        c = st.selectbox("Column", ["foot", "International reputation", "Best position"])
        vc = df[c].value_counts().head(15).rename_axis(c).reset_index(name="count")
        a, b = st.columns(2)
        a.plotly_chart(px.pie(vc, names=c, values="count", hole=.35), width='stretch')
        b.plotly_chart(px.bar(vc, x=c, y="count"), width='stretch')
    elif kind == "Scatter":
        a, b, c3 = st.columns(3)
        x = a.selectbox("X", num_cols, index=num_cols.index("Overall rating"))
        y = b.selectbox("Y", num_cols, index=num_cols.index("Value") if "Value" in num_cols else 1)
        colr = c3.selectbox("Colour", [None, "foot", "Best position", "International reputation"])
        trend = st.checkbox("Trendline (OLS)")
        st.plotly_chart(px.scatter(df, x=x, y=y, color=colr, opacity=.6, hover_name="name",
                                   trendline="ols" if trend else None),
                        width='stretch')
    else:
        m = st.multiselect("Metrics", ["Overall rating", "Potential"], default=["Overall rating", "Potential"])
        g = df.groupby("Age")[m].mean().reset_index()
        st.plotly_chart(px.line(g, x="Age", y=m, markers=True), width='stretch')

# ------------------------------------------------------ sampling & CLT ----
with tabs[3]:
    a, b, c = st.columns(3)
    metric = a.selectbox("Variable", ["Overall rating", "Potential", "Age", "Value", "Wage"])
    n = b.number_input("Sample size", 10, max(10, len(df)), min(100, len(df)))
    seed = c.number_input("Random seed", 0, 10_000, 42)
    d = df.dropna(subset=[metric, "foot"])
    n = int(min(n, len(d)))

    srs = d.sample(n=n, random_state=int(seed))
    props = d["foot"].value_counts(normalize=True)
    per = (props * n).round().astype(int)
    per[per.idxmax()] += n - per.sum()
    strat = pd.concat([d[d["foot"] == f].sample(n=min(int(k), (d["foot"] == f).sum()), random_state=int(seed))
                       for f, k in per.items()])
    srt = d.sort_values(metric).reset_index(drop=True)
    sysm = srt.iloc[:: max(1, len(srt) // n)].head(n)

    comp = pd.DataFrame({
        "Mean": [d[metric].mean(), srs[metric].mean(), strat[metric].mean(), sysm[metric].mean()],
        "Std": [d[metric].std(), srs[metric].std(), strat[metric].std(), sysm[metric].std()],
        "n": [len(d), len(srs), len(strat), len(sysm)],
    }, index=["Population", "Simple random", "Stratified (foot)", "Systematic"])
    comp["Mean diff vs population"] = comp["Mean"] - comp.loc["Population", "Mean"]
    st.dataframe(comp.style.format("{:,.2f}"), width='stretch')

    st.subheader("Central Limit Theorem")
    a, b = st.columns(2)
    sizes = a.multiselect("Sample sizes", [5, 10, 30, 50, 100, 200], default=[10, 30, 100])
    reps = b.slider("Samples per size", 200, 5000, 1000, step=200)
    if sizes:
        pop = d[metric].to_numpy()
        rng = np.random.default_rng(int(seed))
        fig = go.Figure()
        for s in sizes:
            means = rng.choice(pop, size=(reps, s), replace=True).mean(axis=1)
            fig.add_histogram(x=means, name=f"n={s}", opacity=.6, histnorm="probability density")
        fig.add_vline(x=pop.mean(), line_dash="dash", annotation_text="population mean")
        fig.update_layout(barmode="overlay", xaxis_title=f"Sample mean of {metric}")
        st.plotly_chart(fig, width='stretch')
        st.caption("As n grows, the sample means concentrate around the population mean and look more normal.")

# ----------------------------------------------------- hypothesis tests ----
with tabs[4]:
    metric = st.selectbox("Variable", ["Overall rating", "Potential", "Age"], key="ht_var")
    x = df[metric].dropna()
    a, b = st.columns(2)
    conf = a.slider("Confidence level", .80, .99, .95, step=.01)
    mu0 = b.number_input("Null mean (H₀)", value=float(round(x.mean())), step=1.0)
    alpha = 1 - conf
    if len(x) < 3:
        st.warning("Not enough data.")
    else:
        mean, se = x.mean(), stats.sem(x)
        lo, hi = stats.t.interval(conf, len(x) - 1, loc=mean, scale=se)
        z = (mean - mu0) / (x.std() / np.sqrt(len(x)))
        pz = 2 * (1 - stats.norm.cdf(abs(z)))
        t, pt = stats.ttest_1samp(x, mu0)
        st.metric(f"{conf:.0%} CI for mean {metric}", f"{lo:.3f} – {hi:.3f}", f"mean {mean:.3f}", delta_color="off")
        res = pd.DataFrame({"statistic": [z, t], "p-value": [pz, pt],
                            "decision": ["Reject H₀" if p < alpha else "Fail to reject H₀" for p in (pz, pt)]},
                           index=["One-sample Z", "One-sample t"])
        st.dataframe(res.style.format({"statistic": "{:.4f}", "p-value": "{:.4e}"}), width='stretch')
        st.caption(f"H₀: mean {metric} = {mu0}; two-sided, α = {alpha:.2f}.")

# -------------------------------------------------- correlation & anova ----
with tabs[5]:
    st.subheader("Covariance and correlation")
    default = [c for c in ["Overall rating", "Potential", "Age", "Value", "Wage"] if c in num_cols]
    sel = st.multiselect("Variables", num_cols, default=default, key="corr_vars")
    sub = df[sel].dropna() if len(sel) >= 2 else None
    if sub is not None and len(sub) > 2:
        method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
        a, b = st.columns(2)
        a.plotly_chart(px.imshow(sub.corr(method=method), text_auto=".2f", zmin=-1, zmax=1,
                                 color_continuous_scale="RdBu_r", title=f"{method.title()} correlation"),
                       width='stretch')
        b.markdown("**Covariance matrix**")
        b.dataframe(sub.cov().style.format("{:,.2f}"), width='stretch')
        c1, c2 = st.columns(2)
        x = c1.selectbox("Test variable X", sel, index=0)
        y = c2.selectbox("Test variable Y", sel, index=1)
        r, p = (stats.pearsonr if method == "pearson" else stats.spearmanr)(sub[x], sub[y])
        st.info(f"{method.title()} {x} vs {y}: r = {r:.4f}, p = {p:.4e}")
    else:
        st.write("Pick at least two variables.")

    st.subheader("Two-way ANOVA")
    an = df[["Overall rating", "foot", "Age"]].dropna().copy()
    if an["foot"].nunique() > 1 and an["Age"].nunique() >= 3:
        an["Age_Group"] = pd.qcut(an["Age"], q=3, labels=["Young", "Mid", "Senior"], duplicates="drop")
        model = ols('Q("Overall rating") ~ C(foot) * C(Age_Group)', data=an).fit()
        st.dataframe(sm.stats.anova_lm(model, typ=2), width='stretch')
        st.caption("Factors: foot preference × age tercile (Young / Mid / Senior).")
    else:
        st.write("Not enough variation in the filtered data for ANOVA.")

# ------------------------------------------------------ value prediction ----
with tabs[6]:
    st.subheader("Predict market Value")
    feats = st.multiselect("Features", [c for c in num_cols if c != "Value"],
                           default=[c for c in ["Overall rating", "Potential", "Age", "Wage"] if c in num_cols])
    a, b, c = st.columns(3)
    test_size = a.slider("Test size", .1, .5, .2, step=.05)
    seed = b.number_input("Seed", 0, 10_000, 42, key="ml_seed")
    log_t = c.checkbox("Log-transform target", help="Value is heavily right-skewed")
    if st.button("Train models", type="primary") and feats:
        ml = df[feats + ["Value"]].dropna()
        if len(ml) < 30:
            st.warning("Need at least 30 complete rows.")
        else:
            X, y = ml[feats], ml["Value"]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=int(seed))
            models = {
                "Linear": LinearRegression(), "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=1.0, max_iter=10000),
                "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=int(seed)),
                "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=int(seed)),
            }
            out, preds = [], {}
            for name, m in models.items():
                m.fit(Xtr, np.log1p(ytr) if log_t else ytr)
                p = m.predict(Xte)
                p = np.expm1(p) if log_t else p
                preds[name] = p
                out.append({"Model": name, "R²": r2_score(yte, p),
                            "RMSE (€)": np.sqrt(mean_squared_error(yte, p))})
            res = pd.DataFrame(out).set_index("Model")
            st.session_state["ml"] = (res, preds, yte, models, feats)

    if "ml" in st.session_state:
        res, preds, yte, models, used = st.session_state["ml"]
        st.dataframe(res.style.format({"R²": "{:.4f}", "RMSE (€)": "{:,.0f}"}), width='stretch')
        best = res["R²"].idxmax()
        pick = st.selectbox("Actual vs predicted", list(preds), index=list(preds).index(best))
        fig = px.scatter(x=yte, y=preds[pick], labels={"x": "Actual (€)", "y": "Predicted (€)"}, opacity=.5)
        fig.add_shape(type="line", x0=yte.min(), y0=yte.min(), x1=yte.max(), y1=yte.max(),
                      line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, width='stretch')
        m = models[pick]
        if hasattr(m, "feature_importances_"):
            st.plotly_chart(px.bar(x=used, y=m.feature_importances_, labels={"x": "", "y": "importance"},
                                   title=f"{pick} feature importance"), width='stretch')
        elif hasattr(m, "coef_"):
            st.plotly_chart(px.bar(x=used, y=m.coef_, labels={"x": "", "y": "coefficient"},
                                   title=f"{pick} coefficients"), width='stretch')

# ------------------------------------------------------- player explorer ----
with tabs[7]:
    q = st.text_input("Search player name")
    view = df[df["name"].str.contains(q, case=False, na=False)] if q else df
    show = [c for c in ["name", "Team", "Age", "Best position", "foot", "Overall rating", "Potential",
                        "Value", "Wage", "International reputation"] if c in view]
    st.dataframe(view.sort_values("Overall rating", ascending=False)[show].head(200),
                 width='stretch', hide_index=True)
    st.markdown("---")
    names = view["name"].head(500).tolist()
    if names:
        p1 = st.selectbox("Player", names)
        others = st.multiselect("Compare with", [n for n in names if n != p1], max_selections=3)
        axes = ["Pace / Diving", "Shooting / Handling", "Passing / Kicking",
                "Dribbling / Reflexes", "Defending / Pace"]
        axes = [a for a in axes if a in df]
        fig = go.Figure()
        for nm in [p1] + others:
            row = df[df["name"] == nm].iloc[0]
            fig.add_trace(go.Scatterpolar(r=[row[a] for a in axes] + [row[axes[0]]],
                                          theta=axes + [axes[0]], name=nm, fill="toself", opacity=.5))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig, width='stretch')
        st.caption("Six-stat card values; for goalkeepers these columns are Diving/Handling/Kicking/Reflexes/Pace.")
