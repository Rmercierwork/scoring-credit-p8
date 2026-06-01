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


@st.cache_resource(show_spinner="Calcul des SHAP (patience…)")
def build_explainer(_model, X_sample: pd.DataFrame):
    """Explainer SHAP TreeExplainer sur un échantillon."""
    explainer = shap.TreeExplainer(_model)
    return explainer


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


# ── Graphique jauge ──────────────────────────────────────────
def make_gauge(proba: float, seuil: float, decision: str) -> go.Figure:
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
            "text": "Probabilité de défaut<br><span style='font-size:0.8em;color:#757575'>"
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

    # Affichage des principales variables dans un tableau lisible
    info_cols = [c for c in [
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
        "DAYS_BIRTH", "DAYS_EMPLOYED", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
    ] if c in df.columns]

    if info_cols:
        info_data = client_row[info_cols]
        # Mise en forme spéciale pour DAYS_BIRTH → âge
        display = {}
        for col in info_cols:
            val = info_data[col]
            if col == "DAYS_BIRTH" and not pd.isna(val):
                display["Âge (ans)"] = int(abs(val) / 365)
            elif col == "DAYS_EMPLOYED" and not pd.isna(val):
                display["Ancienneté emploi (ans)"] = round(abs(val) / 365, 1)
            else:
                display[col] = round(val, 4) if isinstance(val, float) else val

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

    # ══════════════════════════════════════════════════════════
    # ── SECTION 6 (optionnelle) : Simulation ─────────────────
    # ══════════════════════════════════════════════════════════
    st.divider()
    with st.expander("🧪 Simulation – Modifier les informations client (optionnel)"):
        st.markdown(
            "Modifiez une ou plusieurs valeurs pour simuler un nouveau score sans modifier les données réelles."
        )

        sim_features = [c for c in [
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
        ] if c in feature_cols]

        sim_vals = {}
        cols_sim = st.columns(min(3, len(sim_features)))
        for i, feat in enumerate(sim_features):
            orig = float(client_row[feat]) if not pd.isna(client_row[feat]) else 0.0
            sim_vals[feat] = cols_sim[i % 3].number_input(
                feat, value=orig, format="%.4f", key=f"sim_{feat}",
            )

        if st.button("🔄 Recalculer le score simulé", type="primary"):
            sim_dict = {**features_dict}
            sim_dict.update({k: float(v) for k, v in sim_vals.items()})
            with st.spinner("Appel API en cours…"):
                sim_result = call_api(sim_dict)
            if sim_result:
                sim_proba    = sim_result["probabilite_defaut"]
                sim_decision = sim_result["decision"]
                delta_proba  = sim_proba - proba

                col_s1, col_s2 = st.columns(2)
                col_s1.metric(
                    "Probabilité simulée",
                    f"{round(sim_proba*100,1)} %",
                    delta=f"{delta_proba*100:+.1f} pts vs original",
                    delta_color="inverse",
                )
                icon_sim = "✅" if sim_decision == "ACCORDÉ" else "❌"
                col_s2.metric("Décision simulée", f"{icon_sim} {sim_decision}")

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
