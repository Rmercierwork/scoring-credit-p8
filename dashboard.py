# ============================================================
# Dashboard Streamlit - Scoring Crédit P8
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import os

# ── Configuration de la page ────────────────────────────────
st.set_page_config(
    page_title="Prêt à Dépenser – Dashboard Scoring Crédit",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constantes ───────────────────────────────────────────────
API_URL = "https://rmercierwork-scoring-credit-p7.hf.space"

# Adaptez ces chemins selon votre structure de fichiers
DATA_PATH = "data/application_train_prepared.csv"
MODEL_PATH = "models/lgbm_final.pkl"
SEUIL_PATH = "models/seuil_optimal.pkl"
SCALER_PATH = "models/scaler.pkl"   # StandardScaler du preprocessing P7 (variables continues)

# Variables financières que le chargé de clientèle peut faire varier en simulation.
# Ce sont des leviers "actionnables" (contrairement à l'âge ou aux EXT_SOURCE).
SIM_CANDIDATES = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"]

# ── Accessibilité WCAG – palette accessible daltonisme ───────
COLORS = {
    "accord":   "#2E7D32",  # vert foncé – contraste 7:1 sur blanc
    "refuse":   "#C62828",  # rouge foncé – contraste 7:1 sur blanc
    "neutre":   "#1565C0",  # bleu foncé
    "warning":  "#E65100",  # orange foncé
    "bg_card":  "#F5F5F5",
    "text":     "#212121",
}

# Palette accessible pour graphiques (Okabe-Ito, safe daltonisme)
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#000000"]

# ── CSS personnalisé ─────────────────────────────────────────
st.markdown("""
<style>
/* Critère WCAG 1.4.4 – taille de texte redimensionnable */
html { font-size: 100%; }

/* Critère WCAG 1.4.3 – contrastes */
body, .stMarkdown, .stText { color: #212121 !important; }

.card {
    background: #F5F5F5;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    border-left: 4px solid #1565C0;
}
.card-accord  { border-left-color: #2E7D32; }
.card-refuse  { border-left-color: #C62828; }

.metric-big {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
}
.decision-accord { color: #2E7D32; }
.decision-refuse { color: #C62828; }

/* Focus visible – accessibilité clavier */
button:focus, a:focus, select:focus { outline: 3px solid #1565C0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Chargement des données (mis en cache) ────────────────────
@st.cache_data(show_spinner="Chargement des données clients…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Nettoyage de base : supprimer colonnes 100 % NaN
    df = df.dropna(axis=1, how="all")
    return df


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(model_path: str, seuil_path: str):
    model = joblib.load(model_path)
    seuil = joblib.load(seuil_path)
    return model, seuil


@st.cache_resource(show_spinner="Chargement du scaler…")
def load_scaler(scaler_path: str):
    """Charge le StandardScaler du preprocessing.
    Retourne None si absent : le dashboard reste fonctionnel mais affiche
    les valeurs standardisées et désactive la simulation en euros.
    """
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)


@st.cache_resource(show_spinner="Calcul des SHAP (patience…)")
def build_explainer(_model, X_sample: pd.DataFrame):
    """Explainer SHAP TreeExplainer sur un échantillon."""
    explainer = shap.TreeExplainer(_model)
    return explainer


# ── Conversion euros ↔ z-score via le scaler ─────────────────
class Rescaler:
    """Aller-retour entre valeurs réelles (€, jours…) et valeurs standardisées.

    Le scaler a été entraîné sur un sous-ensemble de variables continues :
    seules celles présentes dans `feature_names_in_` sont convertibles.
    Les variables binaires (ex : CODE_GENDER_F) ne sont pas standardisées et
    sont donc renvoyées telles quelles.
    """
    def __init__(self, scaler):
        self.scaler = scaler
        self.index = {}
        if scaler is not None and hasattr(scaler, "feature_names_in_"):
            self.index = {name: i for i, name in enumerate(scaler.feature_names_in_)}

    def is_scalable(self, feature: str) -> bool:
        return feature in self.index

    def to_real(self, feature: str, z):
        """z-score → valeur réelle."""
        if not self.is_scalable(feature):
            return z
        i = self.index[feature]
        return z * self.scaler.scale_[i] + self.scaler.mean_[i]

    def to_scaled(self, feature: str, real):
        """valeur réelle → z-score."""
        if not self.is_scalable(feature):
            return real
        i = self.index[feature]
        return (real - self.scaler.mean_[i]) / self.scaler.scale_[i]

    def real_series(self, df: pd.DataFrame, feature: str) -> pd.Series:
        """Colonne entière ramenée en valeurs réelles (pour bornes de sliders)."""
        if not self.is_scalable(feature):
            return df[feature]
        i = self.index[feature]
        return df[feature] * self.scaler.scale_[i] + self.scaler.mean_[i]


# ── Appel API ────────────────────────────────────────────────
def call_api(features: dict) -> dict | None:
    """Appelle l'API /predict et retourne le JSON ou None si erreur."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API : {e}", icon="🚨")
        return None


def proba_with_overrides(base_features: dict, overrides_scaled: dict) -> float | None:
    """Renvoie la probabilité de défaut en remplaçant certaines features
    (valeurs déjà standardisées) puis en appelant l'API. Utilisé par la
    simulation et la recherche du seuil de bascule."""
    d = dict(base_features)
    d.update(overrides_scaled)
    res = call_api(d)
    return None if res is None else res["probabilite_defaut"]


def bisect_tipping(pf, lo: float, hi: float, seuil: float,
                   proba_decreasing: bool, n_iter: int = 15):
    """Recherche par dichotomie de la valeur (en €) qui fait basculer la décision.

    pf(x) -> probabilité de défaut pour la valeur réelle x (ou None si erreur API).
    proba_decreasing=True  : augmenter x diminue la proba (ex : revenu).
    proba_decreasing=False : augmenter x augmente la proba (ex : montant du crédit).
    Renvoie la valeur seuil x* (frontière ACCORDÉ / REFUSÉ).
    """
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        p = pf(mid)
        if p is None:
            return None
        passes = p < seuil
        if proba_decreasing:
            # x élevé → proba faible : si ça passe à mid, on peut descendre x
            if passes:
                hi = mid
            else:
                lo = mid
        else:
            # x élevé → proba forte : si ça passe à mid, on peut monter x
            if passes:
                lo = mid
            else:
                hi = mid
    return hi if proba_decreasing else lo


# ── Graphique jauge ──────────────────────────────────────────
def make_gauge(proba: float, seuil: float, decision: str,
               title: str = "Probabilité de défaut") -> go.Figure:
    """Jauge de probabilité de défaut.
    WCAG 1.4.1 : la couleur n'est pas le seul indicateur (texte présent).
    WCAG 1.4.3 : contraste suffisant sur fond blanc.
    """
    color = COLORS["refuse"] if decision == "REFUSÉ" else COLORS["accord"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(proba * 100, 1),
        number={
            "suffix": "%",
            "font": {"size": 36, "color": COLORS["text"]},
        },
        delta={
            "reference": round(seuil * 100, 1),
            "increasing": {"color": COLORS["refuse"]},
            "decreasing": {"color": COLORS["accord"]},
            "suffix": "% vs seuil",
        },
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": COLORS["text"],
                "tickfont": {"size": 12},
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#BDBDBD",
            "steps": [
                {"range": [0, seuil * 100],       "color": "#E8F5E9"},
                {"range": [seuil * 100, 100],      "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": COLORS["warning"], "width": 4},
                "thickness": 0.85,
                "value": round(seuil * 100, 1),
            },
        },
        title={
            "text": f"{title}<br><span style='font-size:0.8em;color:#757575'>"
                    f"Seuil de décision : {round(seuil*100,1)} %</span>",
            "font": {"size": 16},
        },
    ))
    fig.update_layout(
        height=280,
        margin={"t": 60, "b": 10, "l": 30, "r": 30},
        paper_bgcolor="white",
        font_color=COLORS["text"],
    )
    return fig


# ── Graphique SHAP waterfall ─────────────────────────────────
def make_shap_waterfall(shap_values: np.ndarray,
                        feature_names: list[str],
                        base_value: float,
                        top_n: int = 12) -> go.Figure:
    """Top-N features par contribution SHAP (waterfall).
    WCAG 1.4.1 : étiquettes textuelles sur chaque barre.
    """
    idx = np.argsort(np.abs(shap_values))[::-1][:top_n]
    vals  = shap_values[idx]
    names = [feature_names[i] for i in idx]

    colors = [COLORS["refuse"] if v > 0 else COLORS["accord"] for v in vals]

    fig = go.Figure(go.Bar(
        x=vals,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont={"size": 11},
        hovertemplate="%{y} : %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title={
            "text": f"Top {top_n} variables – contribution au score (SHAP)",
            "font": {"size": 14},
        },
        xaxis_title="Valeur SHAP (impact sur la probabilité de défaut)",
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        height=420,
        margin={"l": 200, "r": 80, "t": 50, "b": 50},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color=COLORS["text"],
        xaxis={"zeroline": True, "zerolinecolor": "#757575", "zerolinewidth": 1.5},
    )
    # Ligne de base
    fig.add_vline(x=0, line_width=1.5, line_color="#757575")
    return fig


# ── Graphique distribution d'une feature ────────────────────
def make_distrib(df: pd.DataFrame, feature: str,
                 client_val, decision: str) -> go.Figure:
    """Distribution de la feature pour tous les clients + position du client.
    WCAG 1.4.1 : marqueur client identifié par forme ET couleur.
    """
    series = df[feature].dropna()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if is_numeric:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series,
            nbinsx=40,
            name="Tous les clients",
            marker_color=OKABE_ITO[1],
            opacity=0.75,
            hovertemplate="Valeur : %{x}<br>Nb clients : %{y}<extra></extra>",
        ))
        # Marqueur client – ligne verticale + annotation
        color_client = COLORS["refuse"] if decision == "REFUSÉ" else COLORS["accord"]
        fig.add_vline(
            x=client_val,
            line_width=3,
            line_color=color_client,
            line_dash="dash",
            annotation_text=f"  Client ({client_val:.2f})",
            annotation_font_size=12,
            annotation_font_color=color_client,
        )
        fig.update_layout(
            title={"text": f"Distribution : {feature}", "font": {"size": 13}},
            xaxis_title=feature,
            yaxis_title="Nombre de clients",
            height=320,
            margin={"t": 50, "b": 50, "l": 50, "r": 30},
            paper_bgcolor="white",
            plot_bgcolor="#FAFAFA",
            font_color=COLORS["text"],
            legend={"font": {"size": 11}},
        )
    else:
        counts = series.value_counts().reset_index()
        counts.columns = [feature, "count"]
        fig = px.bar(
            counts, x=feature, y="count",
            color_discrete_sequence=[OKABE_ITO[1]],
            title=f"Distribution : {feature}",
            labels={"count": "Nombre de clients"},
            height=320,
        )
        # Surbrillance de la valeur du client
        fig.update_traces(
            marker_color=[
                COLORS["refuse"] if str(v) == str(client_val) else OKABE_ITO[1]
                for v in counts[feature]
            ]
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="#FAFAFA",
            font_color=COLORS["text"],
            title_font_size=13,
            margin={"t": 50, "b": 50, "l": 50, "r": 30},
        )
    return fig


# ── Graphique bi-varié ───────────────────────────────────────
def make_bivariate(df: pd.DataFrame,
                   feat_x: str, feat_y: str,
                   client_row: pd.Series) -> go.Figure:
    """Scatter bi-varié entre deux features numériques.
    WCAG 1.4.1 : formes différentes pour client vs population.
    """
    sample = df[[feat_x, feat_y]].dropna().sample(
        min(1500, len(df)), random_state=42
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sample[feat_x], y=sample[feat_y],
        mode="markers",
        marker={"color": OKABE_ITO[1], "size": 4, "opacity": 0.5, "symbol": "circle"},
        name="Population",
        hovertemplate=f"{feat_x}: %{{x}}<br>{feat_y}: %{{y}}<extra></extra>",
    ))
    # Point client – forme différente (étoile) + couleur contrastée
    fig.add_trace(go.Scatter(
        x=[client_row[feat_x]], y=[client_row[feat_y]],
        mode="markers+text",
        marker={"color": COLORS["warning"], "size": 18,
                "symbol": "star", "line": {"width": 2, "color": COLORS["text"]}},
        text=["Client sélectionné"],
        textposition="top center",
        textfont={"size": 12, "color": COLORS["text"]},
        name="Client sélectionné",
        hovertemplate=f"{feat_x}: %{{x}}<br>{feat_y}: %{{y}}<extra>Client</extra>",
    ))
    fig.update_layout(
        title={"text": f"Analyse bi-variée : {feat_x} × {feat_y}", "font": {"size": 13}},
        xaxis_title=feat_x,
        yaxis_title=feat_y,
        height=380,
        margin={"t": 50, "b": 60, "l": 60, "r": 30},
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font_color=COLORS["text"],
        legend={"font": {"size": 11}},
    )
    return fig


# ── Mise en forme d'une valeur réelle pour affichage ─────────
def format_real(feature: str, real_val) -> str:
    """Affiche une valeur réelle avec l'unité adaptée."""
    if pd.isna(real_val):
        return "N/A"
    if feature == "DAYS_BIRTH":
        return f"{int(abs(real_val) / 365)} ans"
    if feature in ("DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
                   "DAYS_LAST_PHONE_CHANGE"):
        return f"{abs(real_val) / 365:.1f} ans"
    if feature.startswith("AMT_") or "CREDIT" in feature or "INCOME" in feature:
        return f"{real_val:,.0f} €".replace(",", " ")
    return f"{real_val:.3f}"


# ══════════════════════════════════════════════════════════════
# ── Application principale ───────────────────────────────────
# ══════════════════════════════════════════════════════════════
def main():
    # ── Titre principal (WCAG 2.4.2) ──────────────────────────
    st.title("💳 Dashboard Scoring Crédit – Prêt à Dépenser")
    st.caption(
        "Outil d'aide à la décision à destination des chargés de relation client. "
        "Les décisions sont basées sur un modèle de Machine Learning (LightGBM)."
    )

    # ── Chargement ────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Fichier de données introuvable : `{DATA_PATH}`\n\n"
            "Vérifiez que le fichier CSV est bien présent dans le dossier `data/`.",
            icon="🚨",
        )
        st.stop()

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Modèle introuvable : `{MODEL_PATH}`\n\n"
            "Vérifiez que `lgbm_final.pkl` est bien dans le dossier `models/`.",
            icon="🚨",
        )
        st.stop()

    df    = load_data(DATA_PATH)
    model, seuil = load_model(MODEL_PATH, SEUIL_PATH)
    scaler = load_scaler(SCALER_PATH)
    rescaler = Rescaler(scaler)

    # Colonnes features (tout sauf TARGET si présente)
    feature_cols = [c for c in df.columns if c not in ("TARGET", "SK_ID_CURR")]
    num_features = [c for c in feature_cols
                    if pd.api.types.is_numeric_dtype(df[c])]

    # ── Sidebar ───────────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 Sélection du client")

        # Sélection par ID ou par index
        if "SK_ID_CURR" in df.columns:
            id_list = sorted(df["SK_ID_CURR"].dropna().astype(int).unique().tolist())
            client_id = st.selectbox(
                "Identifiant client (SK_ID_CURR)",
                options=id_list,
                help="Sélectionnez l'identifiant du client à analyser.",
            )
            client_row = df[df["SK_ID_CURR"] == client_id].iloc[0]
        else:
            idx = st.number_input(
                "Index du client (0 – {})".format(len(df) - 1),
                min_value=0, max_value=len(df) - 1,
                value=0, step=1,
            )
            client_row = df.iloc[idx]

        st.divider()
        st.header("⚙️ Options d'affichage")
        top_n_shap = st.slider(
            "Nombre de variables SHAP affichées", 5, 20, 12, step=1,
        )

        st.divider()
        if scaler is None:
            st.warning(
                "Scaler non trouvé (`models/scaler.pkl`) : les valeurs sont "
                "affichées standardisées et la simulation en euros est désactivée.",
                icon="⚠️",
            )
        st.caption(
            "ℹ️ Ce dashboard est un outil d'aide à la décision. "
            "La décision finale reste sous la responsabilité du chargé de relation client."
        )

    # ── Appel API ────────────────────────────────────────────
    features_dict = client_row[feature_cols].to_dict()
    # Convertir numpy types → python natifs pour JSON
    features_dict = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in features_dict.items()
    }

    with st.spinner("Appel de l'API de scoring…"):
        result = call_api(features_dict)

    if result is None:
        st.warning("Impossible de contacter l'API. Vérifiez votre connexion.", icon="⚠️")
        st.stop()

    proba    = result["probabilite_defaut"]
    seuil_v  = result["seuil"]
    decision = result["decision"]

    # ══════════════════════════════════════════════════════════
    # ── CARTES DE SYNTHÈSE (en-tête) ─────────────────────────
    # ══════════════════════════════════════════════════════════
    # Coup d'œil rapide : décision, proba, seuil, marge, et deux
    # repères client en unités réelles si le scaler est disponible.
    ecart_pts = (proba - seuil_v) * 100
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Décision", f"{'✅' if decision == 'ACCORDÉ' else '❌'} {decision}")
    k2.metric("Probabilité de défaut", f"{proba*100:.1f} %")
    k3.metric("Seuil de décision", f"{seuil_v*100:.1f} %")
    k4.metric("Marge au seuil", f"{ecart_pts:+.1f} pts",
              help="Négatif = sous le seuil (favorable). Positif = au-dessus (défavorable).")

    if rescaler.is_scalable("AMT_INCOME_TOTAL"):
        revenu = rescaler.to_real("AMT_INCOME_TOTAL", client_row["AMT_INCOME_TOTAL"])
        k5.metric("Revenu annuel", format_real("AMT_INCOME_TOTAL", revenu))
    if rescaler.is_scalable("DAYS_BIRTH") and "DAYS_BIRTH" in df.columns:
        age = rescaler.to_real("DAYS_BIRTH", client_row["DAYS_BIRTH"])
        k6.metric("Âge", format_real("DAYS_BIRTH", age))

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ── SECTION 1 : Score & Décision ─────────────────────────
    # ══════════════════════════════════════════════════════════
    st.subheader("📊 Score et décision de crédit")

    col_gauge, col_decision = st.columns([1.6, 1], gap="large")

    with col_gauge:
        fig_gauge = make_gauge(proba, seuil_v, decision)
        st.plotly_chart(
            fig_gauge, use_container_width=True,
            config={"displayModeBar": False},
        )
        # WCAG 1.1.1 – alternative textuelle à la jauge
        st.caption(
            f"ℹ️ Texte alternatif : Probabilité de défaut de {round(proba*100,1)} %, "
            f"seuil à {round(seuil_v*100,1)} %. Décision : {decision}."
        )

    with col_decision:
        css_class = "card-accord" if decision == "ACCORDÉ" else "card-refuse"
        dec_class  = "decision-accord" if decision == "ACCORDÉ" else "decision-refuse"
        icon       = "✅" if decision == "ACCORDÉ" else "❌"
        st.markdown(f"""
        <div class="card {css_class}" role="region" aria-label="Résultat de la décision">
            <div class="metric-big {dec_class}">{icon} {decision}</div>
            <br>
            <strong>Probabilité de défaut :</strong> {round(proba*100,1)} %<br>
            <strong>Seuil de décision :</strong> {round(seuil_v*100,1)} %<br>
            <strong>Écart au seuil :</strong> {round((proba - seuil_v)*100, 1):+.1f} pts
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Comment lire ce résultat ?**")
        if decision == "ACCORDÉ":
            st.success(
                f"La probabilité de défaut ({round(proba*100,1)} %) est **inférieure** "
                f"au seuil ({round(seuil_v*100,1)} %). Le crédit est accordé.",
                icon="✅",
            )
        else:
            st.error(
                f"La probabilité de défaut ({round(proba*100,1)} %) est **supérieure** "
                f"au seuil ({round(seuil_v*100,1)} %). Le crédit est refusé.",
                icon="❌",
            )

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ── SECTION 2 : Explication SHAP ─────────────────────────
    # ══════════════════════════════════════════════════════════
    st.subheader("🔬 Explication de la décision (SHAP)")
    st.markdown(
        "Les barres rouges représentent les variables qui **augmentent** la probabilité de défaut "
        "(défavorables). Les barres vertes la **diminuent** (favorables)."
    )

    explainer  = build_explainer(model, df[feature_cols].head(500))
    client_arr = client_row[feature_cols].values.reshape(1, -1)
    shap_vals  = explainer.shap_values(client_arr)

    # LightGBM binaire → shap_values peut être liste [class0, class1]
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
        bv = explainer.expected_value[1]
    else:
        sv = shap_vals[0]
        bv = explainer.expected_value

    fig_shap = make_shap_waterfall(sv, feature_cols, bv, top_n=top_n_shap)
    st.plotly_chart(fig_shap, use_container_width=True)

    # WCAG 1.1.1 – alternative textuelle SHAP
    top3_idx  = np.argsort(np.abs(sv))[::-1][:3]
    top3_str  = ", ".join(
        f"{feature_cols[i]} ({sv[i]:+.3f})" for i in top3_idx
    )
    st.caption(f"ℹ️ Variables les plus influentes : {top3_str}.")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ── SECTION 3 : Profil du client ─────────────────────────
    # ══════════════════════════════════════════════════════════
    st.subheader("👤 Informations descriptives du client")
    if scaler is not None:
        st.caption(
            "Valeurs ramenées en unités réelles (euros, années) via le scaler du preprocessing."
        )

    # Affichage des principales variables dans un tableau lisible
    info_cols = [c for c in [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
    ] if c in df.columns]

    labels = {
        "AMT_INCOME_TOTAL": "Revenu annuel",
        "AMT_CREDIT":       "Montant du crédit",
        "AMT_ANNUITY":      "Mensualité (annuité)",
        "DAYS_BIRTH":       "Âge",
        "DAYS_EMPLOYED":    "Ancienneté emploi",
        "EXT_SOURCE_1":     "Score externe 1",
        "EXT_SOURCE_2":     "Score externe 2",
        "EXT_SOURCE_3":     "Score externe 3",
    }

    if info_cols:
        display = {}
        for col in info_cols:
            z = client_row[col]
            # On repasse en réel si la variable est standardisée, sinon valeur brute
            real = rescaler.to_real(col, z) if rescaler.is_scalable(col) else z
            display[labels.get(col, col)] = format_real(col, real)

        info_df = pd.DataFrame(
            {"Variable": list(display.keys()),
             "Valeur client": list(display.values())}
        )
        st.dataframe(
            info_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Variable":      st.column_config.TextColumn("Variable"),
                "Valeur client": st.column_config.TextColumn("Valeur client"),
            },
        )
    else:
        st.info("Variables descriptives standard non trouvées dans le dataset.")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ── SECTION 4 : Comparaison client / population ──────────
    # ══════════════════════════════════════════════════════════
    st.subheader("📈 Comparaison client / population")
    st.markdown("Sélectionnez une variable pour voir où se situe le client par rapport à l'ensemble des clients.")

    feat_distrib = st.selectbox(
        "Variable à visualiser",
        options=feature_cols,
        index=feature_cols.index("EXT_SOURCE_2") if "EXT_SOURCE_2" in feature_cols else 0,
        help="Choisissez n'importe quelle variable du modèle.",
    )

    client_val = client_row[feat_distrib]
    fig_dist   = make_distrib(df, feat_distrib, client_val, decision)
    st.plotly_chart(fig_dist, use_container_width=True)

    # Statistiques résumées
    series = df[feat_distrib].dropna()
    if pd.api.types.is_numeric_dtype(series):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Valeur client",  f"{client_val:.3f}" if not pd.isna(client_val) else "N/A")
        c2.metric("Médiane (pop.)", f"{series.median():.3f}")
        c3.metric("Moyenne (pop.)", f"{series.mean():.3f}")
        c4.metric("Écart-type",     f"{series.std():.3f}")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ── SECTION 5 : Analyse bi-variée ────────────────────────
    # ══════════════════════════════════════════════════════════
    st.subheader("🔀 Analyse bi-variée")
    st.markdown("Comparez deux variables numériques et visualisez la position du client dans la population.")

    col_biv1, col_biv2 = st.columns(2)
    with col_biv1:
        default_x = num_features.index("EXT_SOURCE_2") if "EXT_SOURCE_2" in num_features else 0
        feat_x = st.selectbox("Variable X", options=num_features, index=default_x, key="biv_x")
    with col_biv2:
        default_y = num_features.index("EXT_SOURCE_3") if "EXT_SOURCE_3" in num_features else 1
        feat_y = st.selectbox("Variable Y", options=num_features, index=default_y, key="biv_y")

    if feat_x != feat_y:
        fig_biv = make_bivariate(df, feat_x, feat_y, client_row)
        st.plotly_chart(fig_biv, use_container_width=True)
    else:
        st.warning("Sélectionnez deux variables différentes.", icon="⚠️")

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ── SECTION 6 : Simulation « et si… ? » ──────────────────
    # ══════════════════════════════════════════════════════════
    st.subheader("🧪 Simulation « et si… ? »")

    # Leviers réellement présents dans le modèle ET convertibles en euros
    sim_features = [c for c in SIM_CANDIDATES
                    if c in feature_cols and rescaler.is_scalable(c)]

    if scaler is None or not sim_features:
        st.info(
            "La simulation en euros nécessite le scaler (`models/scaler.pkl`) et au moins "
            "une variable financière parmi le revenu, le montant du crédit ou la mensualité.",
            icon="ℹ️",
        )
    else:
        st.markdown(
            "Modifiez les informations financières du client pour recalculer le score, "
            "sans toucher aux données réelles. Utile pour répondre à *« et si le revenu "
            "augmentait ? »* ou *« et avec un montant de crédit plus faible ? »*."
        )

        labels_sim = {
            "AMT_INCOME_TOTAL": "Revenu annuel (€)",
            "AMT_CREDIT":       "Montant du crédit (€)",
            "AMT_ANNUITY":      "Mensualité / annuité (€)",
        }

        # Un slider par levier, borné sur la distribution réelle (1er–99e centile)
        sim_real = {}
        cols_sliders = st.columns(len(sim_features))
        for col_widget, feat in zip(cols_sliders, sim_features):
            real_col = rescaler.real_series(df, feat)
            lo = float(max(0.0, np.nanpercentile(real_col, 1)))
            hi = float(np.nanpercentile(real_col, 99))
            current = float(rescaler.to_real(feat, client_row[feat]))
            current = min(max(current, lo), hi)  # borne au cas où le client est un outlier
            step = max(1000.0, round((hi - lo) / 100, -2))
            with col_widget:
                sim_real[feat] = st.slider(
                    labels_sim.get(feat, feat),
                    min_value=round(lo, -2), max_value=round(hi, -2),
                    value=round(current, -2), step=step,
                    help=f"Valeur actuelle du client : {current:,.0f} €".replace(",", " "),
                )

        if st.button("🔄 Recalculer le score simulé", type="primary"):
            overrides = {f: rescaler.to_scaled(f, v) for f, v in sim_real.items()}
            with st.spinner("Appel API en cours…"):
                sim_res = call_api({**features_dict, **overrides})
            if sim_res:
                sim_proba    = sim_res["probabilite_defaut"]
                sim_decision = sim_res["decision"]
                delta_proba  = (sim_proba - proba) * 100

                cs1, cs2, cs3 = st.columns(3)
                cs1.metric("Probabilité simulée", f"{sim_proba*100:.1f} %",
                           delta=f"{delta_proba:+.1f} pts vs actuel",
                           delta_color="inverse")
                icon_sim = "✅" if sim_decision == "ACCORDÉ" else "❌"
                cs2.metric("Décision simulée", f"{icon_sim} {sim_decision}")
                cs3.metric("Décision initiale",
                           f"{'✅' if decision == 'ACCORDÉ' else '❌'} {decision}")

                # Message de bascule explicite
                if sim_decision != decision:
                    if sim_decision == "ACCORDÉ":
                        st.success(
                            "Avec ces paramètres, la décision **basculerait de REFUSÉ à ACCORDÉ**.",
                            icon="✅",
                        )
                    else:
                        st.error(
                            "Avec ces paramètres, la décision **basculerait de ACCORDÉ à REFUSÉ**.",
                            icon="❌",
                        )
                else:
                    st.info("La décision reste inchangée avec ces paramètres.", icon="ℹ️")

        # ── Recherche du seuil de bascule ─────────────────────
        st.markdown("---")
        st.markdown("**🎯 Seuil de bascule**")
        st.markdown(
            "Cherche automatiquement la valeur qui ferait changer la décision, "
            "toutes choses égales par ailleurs."
        )

        # Options possibles selon les leviers disponibles et le sens métier
        lever_options = {}
        if "AMT_INCOME_TOTAL" in sim_features:
            lever_options["Revenu à atteindre pour être ACCORDÉ"] = ("AMT_INCOME_TOTAL", True)
        if "AMT_CREDIT" in sim_features:
            lever_options["Montant de crédit maximal pour être ACCORDÉ"] = ("AMT_CREDIT", False)

        if lever_options:
            choix = st.selectbox("Levier à analyser", options=list(lever_options.keys()))
            feat_t, decreasing = lever_options[choix]

            if st.button("Calculer le seuil de bascule"):
                if decision == "ACCORDÉ":
                    st.info("Ce dossier est déjà accordé : pas de bascule à rechercher.", icon="ℹ️")
                else:
                    real_col = rescaler.real_series(df, feat_t)
                    lo = float(max(0.0, np.nanpercentile(real_col, 1)))
                    hi = float(np.nanpercentile(real_col, 99))
                    current = float(rescaler.to_real(feat_t, client_row[feat_t]))

                    # Fonction proba en fonction de la valeur réelle du levier
                    def pf(x_real, _feat=feat_t):
                        return proba_with_overrides(
                            features_dict, {_feat: rescaler.to_scaled(_feat, x_real)}
                        )

                    with st.spinner("Recherche par dichotomie (quelques appels API)…"):
                        if decreasing:
                            # Revenu : borne haute = meilleur cas
                            search_hi = max(hi, current * 3)
                            if (pf(search_hi) or 1.0) >= seuil_v:
                                st.warning(
                                    "Même à un revenu très élevé, ce dossier ne passe pas : "
                                    "d'autres facteurs (scores externes, historique) dominent la décision.",
                                    icon="⚠️",
                                )
                                x_star = None
                            else:
                                x_star = bisect_tipping(pf, current, search_hi, seuil_v,
                                                        proba_decreasing=True)
                        else:
                            # Crédit : borne basse = meilleur cas
                            search_lo = min(lo, current * 0.2)
                            if (pf(search_lo) or 1.0) >= seuil_v:
                                st.warning(
                                    "Même avec un crédit très faible, ce dossier ne passe pas : "
                                    "d'autres facteurs dominent la décision.",
                                    icon="⚠️",
                                )
                                x_star = None
                            else:
                                x_star = bisect_tipping(pf, search_lo, current, seuil_v,
                                                        proba_decreasing=False)

                    if x_star is not None:
                        p_star = pf(x_star)
                        if decreasing:
                            delta = x_star - current
                            st.success(
                                f"Il faudrait un **revenu d'environ {x_star:,.0f} €** "
                                f"(soit **+{delta:,.0f} €** par rapport aux {current:,.0f} € actuels) "
                                f"pour que le dossier passe (proba ≈ {p_star*100:.1f} %).".replace(",", " "),
                                icon="🎯",
                            )
                        else:
                            delta = current - x_star
                            st.success(
                                f"Il faudrait un **crédit d'au plus {x_star:,.0f} €** "
                                f"(soit **−{delta:,.0f} €** par rapport aux {current:,.0f} € demandés) "
                                f"pour que le dossier passe (proba ≈ {p_star*100:.1f} %).".replace(",", " "),
                                icon="🎯",
                            )

    # ── Pied de page ─────────────────────────────────────────
    st.divider()
    st.caption(
        "Dashboard P8 – Prêt à Dépenser | "
        "Modèle : LightGBM | "
        "Conformité WCAG : 1.1.1, 1.4.1, 1.4.3, 1.4.4, 2.4.2 | "
        "Les décisions restent sous la responsabilité des chargés de relation client."
    )


if __name__ == "__main__":
    main()
