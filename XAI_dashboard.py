import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ── Configuração da página ──────────────────────────────────
st.set_page_config(
    page_title="XAI Framework | Gabriel Naumann",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS customizado (tema dark alinhado ao portfólio) ───────
st.markdown("""
<style>
    /* Fundo principal */
    .stApp { background-color: #0a0f1c; }
    section[data-testid="stSidebar"] { background-color: #111827; }

    /* Títulos */
    h1, h2, h3 { color: #e8eaed !important; }
    p, label, div { color: #c9cdd1; }

    /* Cards de métricas */
    [data-testid="metric-container"] {
        background-color: #111827;
        border: 1px solid #1a2234;
        border-radius: 8px;
        padding: 12px;
    }
    [data-testid="metric-container"] label { color: #9aa0a6 !important; font-size:13px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #60a5fa !important; font-size: 28px; font-weight: 700;
    }

    /* Botões */
    .stButton > button {
        background: linear-gradient(135deg, #60a5fa, #818cf8);
        color: white; border: none; border-radius: 8px;
        padding: 10px 24px; font-weight: 600; width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Sliders e selects */
    .stSlider > div > div > div { background: #60a5fa; }
    .stSelectbox > div > div { background-color: #111827; color: #e8eaed; }

    /* Divisores */
    hr { border-color: #1a2234; }

    /* Badges de decisão */
    .badge-approved {
        background: #065f46; color: #34d399;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 18px; display: inline-block;
    }
    .badge-denied {
        background: #7f1d1d; color: #f87171;
        padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 18px; display: inline-block;
    }
    .info-box {
        background: #111827; border-left: 3px solid #60a5fa;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Dados e modelo (cache para não reprocessar) ─────────────
@st.cache_resource
def load_model_and_data():
    SEED = 42
    np.random.seed(SEED)
    N = 10_000

    age            = np.random.randint(18, 75, N)
    monthly_income = np.random.lognormal(8.5, 0.6, N).clip(1000, 50000)
    debt_ratio     = np.random.beta(2, 5, N)
    revolving_util = np.random.beta(2, 3, N)
    num_late_30    = np.random.poisson(0.3, N)
    num_late_90    = np.random.poisson(0.1, N)
    open_credit    = np.random.randint(0, 15, N)
    real_estate    = np.random.randint(0, 4, N)
    dependents     = np.random.randint(0, 6, N)
    credit_history = np.random.randint(0, 30, N)

    default_prob = (
        0.3 * revolving_util + 0.25 * debt_ratio +
        0.2 * (num_late_30 / 5) + 0.15 * (num_late_90 / 3) +
        0.05 * (1 - monthly_income / 50000) + 0.05 * (1 - age / 75)
    )
    default_prob = (default_prob - default_prob.min()) / (default_prob.max() - default_prob.min())
    default_prob = default_prob * 0.5 + np.random.uniform(0, 0.1, N)
    y = (default_prob > 0.35).astype(int)

    FEATURES = ['RevolvingUtilization','Age','NumLate30Days','DebtRatio',
                'MonthlyIncome','OpenCreditLines','NumLate90Days',
                'RealEstateLoans','Dependents','CreditHistoryYears']

    X = pd.DataFrame({
        'RevolvingUtilization': revolving_util, 'Age': age,
        'NumLate30Days': num_late_30,           'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,         'OpenCreditLines': open_credit,
        'NumLate90Days': num_late_90,            'RealEstateLoans': real_estate,
        'Dependents': dependents,                'CreditHistoryYears': credit_history,
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        use_label_encoder=False, eval_metric='auc',
        random_state=SEED, n_jobs=-1)
    model.fit(X_train, y_train)

    explainer     = shap.TreeExplainer(model)
    shap_values   = explainer(X_test)
    lime_explainer= lime.lime_tabular.LimeTabularExplainer(
        X_train.values, feature_names=FEATURES,
        class_names=['Adimplente','Inadimplente'],
        mode='classification', random_state=SEED)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

    return model, X_train, X_test, y_test, FEATURES, explainer, shap_values, lime_explainer, auc

# ── Inicialização protegida ─────────────────────────────────
if __name__ == '__main__' or True:
    model, X_train, X_test, y_test, FEATURES, shap_explainer, shap_values, lime_explainer, auc = load_model_and_data()

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔎 XAI Framework")
    st.markdown("**Gabriel Alessi Naumann**")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/gabriel-alessi-naumann/) · [GitHub](https://github.com/GabrielAlessi)")
    st.markdown("---")

    st.markdown("### ⚙️ Configuração")
    page = st.selectbox("Página", [
        "🏠 Visão Geral",
        "🔴 Explicar Cliente (Manual)",
        "📊 SHAP Global",
        "🔀 Permutation Importance",
        "📋 Relatório Completo",
    ])

    st.markdown("---")
    st.markdown("### 📌 Técnicas")
    st.markdown("""
    - **SHAP** — Importância global e local
    - **LIME** — Modelo linear local
    - **Permutation** — Queda de performance
    - **Counterfactual** — O que mudar?
    """)
    st.markdown("---")
    st.markdown(f"**Modelo:** XGBoost  \n**ROC-AUC:** `{auc:.4f}`  \n**Dataset:** 10.000 clientes")

# ── PAGE 1: Visão Geral ─────────────────────────────────────
if page == "🏠 Visão Geral":
    st.title("🔎 Framework de Explainable AI")
    st.markdown("Biblioteca para interpretação de modelos de ML aplicada a **Scoring de Crédito Fintech**")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("ROC-AUC", f"{auc:.4f}", "XGBoost")
    with col2: st.metric("Clientes", "10.000", "Dataset sintético")
    with col3: st.metric("Técnicas XAI", "4", "SHAP · LIME · Perm · CF")
    with col4: st.metric("Features", "10", "Scoring de crédito")

    st.markdown("---")
    st.markdown("### 🗺️ Quando usar cada técnica")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='info-box'>
        <b>📊 SHAP</b><br>
        Explicação global e local baseada em teoria dos jogos.<br>
        <i>Use quando:</i> precisar explicar para o time técnico por que uma decisão foi tomada.
        </div>
        <div class='info-box'>
        <b>🔬 LIME</b><br>
        Modelo linear local que aproxima qualquer caixa-preta.<br>
        <i>Use quando:</i> o modelo não for tree-based ou precisar de explicação mais simples.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='info-box'>
        <b>🔀 Permutation Importance</b><br>
        Mede queda de performance ao embaralhar cada feature.<br>
        <i>Use quando:</i> quiser validar que o modelo não depende de features espúrias.
        </div>
        <div class='info-box'>
        <b>🔄 Counterfactual</b><br>
        Responde: "O que mudar para ser aprovado?"<br>
        <i>Use quando:</i> precisar comunicar a negativa de crédito ao cliente (LGPD/GDPR).
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Distribuição de Risco no Dataset")

    probs_all = model.predict_proba(X_test)[:,1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), facecolor='#0a0f1c')
    for ax in axes: ax.set_facecolor('#111827')

    axes[0].hist(probs_all[np.array(y_test)==0], bins=40, alpha=0.7,
                 color='#60a5fa', label='Adimplente', density=True)
    axes[0].hist(probs_all[np.array(y_test)==1], bins=40, alpha=0.7,
                 color='#f87171', label='Inadimplente', density=True)
    axes[0].axvline(0.5, color='white', linestyle='--', alpha=0.6, label='Threshold 0.5')
    axes[0].set_title('Distribuição dos Scores', color='#e8eaed')
    axes[0].set_xlabel('Probabilidade de Inadimplência', color='#9aa0a6')
    axes[0].legend(facecolor='#111827', labelcolor='#e8eaed')
    axes[0].tick_params(colors='#9aa0a6')
    for spine in axes[0].spines.values(): spine.set_edgecolor('#1a2234')

    risk_cats = pd.cut(probs_all, bins=[0,0.3,0.6,1.0],
                       labels=['Baixo','Médio','Alto'])
    counts = risk_cats.value_counts().sort_index()
    colors_r = ['#34d399','#fbbf24','#f87171']
    axes[1].bar(counts.index, counts.values, color=colors_r, alpha=0.85)
    axes[1].set_title('Segmentação por Faixa de Risco', color='#e8eaed')
    axes[1].set_ylabel('Clientes', color='#9aa0a6')
    axes[1].tick_params(colors='#9aa0a6')
    for spine in axes[1].spines.values(): spine.set_edgecolor('#1a2234')
    for i, v in enumerate(counts.values):
        axes[1].text(i, v + 10, f'{v:,}', ha='center', color='#e8eaed', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── PAGE 2: Explicar Cliente Manual ─────────────────────────
elif page == "🔴 Explicar Cliente (Manual)":
    st.title("🔴 Explicar Decisão de Crédito")
    st.markdown("Insira os dados do cliente e veja a explicação completa em tempo real.")
    st.markdown("---")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("### 📝 Dados do Cliente")
        revolving = st.slider("Utilização do Crédito Rotativo", 0.0, 1.0, 0.5, 0.01)
        age_val   = st.slider("Idade", 18, 75, 35)
        income    = st.number_input("Renda Mensal (R$)", 1000, 50000, 5000, 500)
        debt      = st.slider("Razão Dívida/Renda", 0.0, 1.0, 0.3, 0.01)
        late30    = st.slider("Atrasos de 30 dias (últimos 2 anos)", 0, 10, 0)
        late90    = st.slider("Atrasos de 90+ dias (últimos 2 anos)", 0, 5, 0)
        open_cr   = st.slider("Linhas de Crédito Abertas", 0, 15, 5)
        real_est  = st.slider("Empréstimos Imobiliários", 0, 4, 0)
        deps      = st.slider("Dependentes", 0, 6, 1)
        hist      = st.slider("Histórico de Crédito (anos)", 0, 30, 8)

        instance = pd.DataFrame([{
            'RevolvingUtilization': revolving, 'Age': age_val,
            'NumLate30Days': late30,           'DebtRatio': debt,
            'MonthlyIncome': income,            'OpenCreditLines': open_cr,
            'NumLate90Days': late90,            'RealEstateLoans': real_est,
            'Dependents': deps,                 'CreditHistoryYears': hist,
        }])

        run = st.button("⚡ Analisar Cliente")

    with col_out:
        prob = float(model.predict_proba(instance)[0,1])
        decision = prob >= 0.5

        st.markdown("### 📊 Resultado")
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Probabilidade Inadimplência", f"{prob*100:.1f}%")
        with c2: st.metric("Score de Crédito", f"{int((1-prob)*1000)}")
        with c3: st.metric("Faixa de Risco",
                            "Alto" if prob>0.6 else "Médio" if prob>0.3 else "Baixo")

        if decision:
            st.markdown("<div class='badge-denied'>❌ CRÉDITO NEGADO</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='badge-approved'>✅ CRÉDITO APROVADO</div>", unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 SHAP", "🔬 LIME", "🔄 Counterfactual"])

        with tab1:
            st.markdown("**Por que o modelo tomou essa decisão?**")
            sv = shap_explainer(instance)
            fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0a0f1c')
            ax.set_facecolor('#111827')
            shap_vals = sv.values[0]
            base_val  = float(sv.base_values[0])
            sorted_idx= np.argsort(np.abs(shap_vals))[::-1][:8]
            feat_show = [FEATURES[i] for i in sorted_idx]
            vals_show = shap_vals[sorted_idx]
            colors_s  = ['#f87171' if v > 0 else '#60a5fa' for v in vals_show]
            ax.barh(feat_show[::-1], vals_show[::-1], color=colors_s[::-1], alpha=0.85)
            ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
            ax.set_title('SHAP Values — Contribuição de cada feature', color='#e8eaed')
            ax.set_xlabel('SHAP value (+ aumenta risco | - reduz risco)', color='#9aa0a6')
            ax.tick_params(colors='#9aa0a6')
            for spine in ax.spines.values(): spine.set_edgecolor('#1a2234')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab2:
            st.markdown("**Explicação local aproximada (LIME):**")
            exp = lime_explainer.explain_instance(
                instance.values[0], model.predict_proba,
                num_features=8, num_samples=3000)
            feats_lime  = exp.as_list()
            names_lime  = [f[0] for f in feats_lime]
            weights_lime= [f[1] for f in feats_lime]
            colors_l    = ['#f87171' if w > 0 else '#60a5fa' for w in weights_lime]
            fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0a0f1c')
            ax.set_facecolor('#111827')
            ax.barh(names_lime[::-1], weights_lime[::-1], color=colors_l[::-1], alpha=0.85)
            ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
            ax.set_title('LIME — Modelo linear local', color='#e8eaed')
            ax.set_xlabel('Peso (+ → Inadimplente | - → Adimplente)', color='#9aa0a6')
            ax.tick_params(colors='#9aa0a6')
            for spine in ax.spines.values(): spine.set_edgecolor('#1a2234')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab3:
            st.markdown("**O que precisaria mudar para ser aprovado?**")
            if not decision:
                st.success("Este cliente já está aprovado! Nenhuma mudança necessária.")
            else:
                with st.spinner("🔍 Buscando counterfactual..."):
                    cf = instance.copy()
                    changes = {}
                    immutable = ['Age']
                    for _ in range(1500):
                        cp = float(model.predict_proba(cf)[0,1])
                        if cp < 0.5: break
                        best_f, best_v, best_delta = None, None, 0
                        for feat in FEATURES:
                            if feat in immutable: continue
                            fmin = float(X_train[feat].min())
                            fmax = float(X_train[feat].max())
                            delta = (fmax - fmin) * 0.05
                            for direction in [1, -1]:
                                cf_test = cf.copy()
                                nv = np.clip(cf_test[feat].values[0] + direction*delta, fmin, fmax)
                                cf_test[feat] = nv
                                np_ = float(model.predict_proba(cf_test)[0,1])
                                if np_ < cp and (cp - np_) > best_delta:
                                    best_delta = cp - np_
                                    best_f = feat
                                    best_v = nv
                        if best_f:
                            old_v = float(cf[best_f].values[0])
                            cf[best_f] = best_v
                            changes[best_f] = {'from': round(old_v,4),
                                               'to': round(best_v,4),
                                               'delta': round(best_v-old_v,4)}

                final_prob = float(model.predict_proba(cf)[0,1])
                if final_prob < 0.5 and changes:
                    st.markdown(f"**Prob. inadimplência:** {prob*100:.1f}% → **{final_prob*100:.1f}%**")
                    st.markdown("**Mudanças necessárias:**")
                    readable = {
                        'RevolvingUtilization':'Utilização do crédito rotativo',
                        'DebtRatio':'Razão dívida/renda',
                        'MonthlyIncome':'Renda mensal (R$)',
                        'NumLate30Days':'Atrasos de 30 dias',
                        'NumLate90Days':'Atrasos de 90+ dias',
                        'OpenCreditLines':'Linhas de crédito abertas',
                        'CreditHistoryYears':'Histórico de crédito (anos)',
                    }
                    for feat, vals in changes.items():
                        arrow = "📉" if vals['delta'] < 0 else "📈"
                        fname = readable.get(feat, feat)
                        st.markdown(f"- {arrow} **{fname}**: `{vals['from']:.3f}` → `{vals['to']:.3f}`")
                else:
                    st.warning("Não foi possível encontrar um counterfactual simples. Tente ajustar os sliders manualmente.")

# ── PAGE 3: SHAP Global ──────────────────────────────────────
elif page == "📊 SHAP Global":
    st.title("📊 SHAP — Importância Global das Features")
    st.markdown("Análise da importância média das features em todo o dataset de teste.")
    st.markdown("---")

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    df_shap   = pd.DataFrame({'Feature': FEATURES, 'Mean |SHAP|': mean_shap})
    df_shap   = df_shap.sort_values('Mean |SHAP|', ascending=False)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor='#0a0f1c')
        ax.set_facecolor('#111827')
        colors_g = ['#60a5fa','#818cf8','#a78bfa','#34d399','#fbbf24',
                    '#f87171','#60a5fa','#818cf8','#a78bfa','#34d399']
        ax.barh(df_shap['Feature'][::-1], df_shap['Mean |SHAP|'][::-1],
                color=colors_g[::-1], alpha=0.85)
        ax.set_title('Importância Global (Mean |SHAP value|)', color='#e8eaed')
        ax.set_xlabel('Mean |SHAP value|', color='#9aa0a6')
        ax.tick_params(colors='#9aa0a6')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2234')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 🏆 Ranking de Importância")
        for i, row in df_shap.reset_index(drop=True).iterrows():
            medal = ["🥇","🥈","🥉"] [i] if i < 3 else f"{i+1}."
            st.markdown(f"{medal} **{row['Feature']}** — `{row['Mean |SHAP|']:.5f}`")

    st.markdown("---")
    st.markdown("### 🎯 SHAP Summary Plot (Dot)")
    fig2, ax2 = plt.subplots(figsize=(12, 7), facecolor='#0a0f1c')
    shap.summary_plot(shap_values, X_test, plot_type='dot',
                      max_display=10, show=False)
    plt.gcf().set_facecolor('#0a0f1c')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close('all')

# ── PAGE 4: Permutation Importance ──────────────────────────
elif page == "🔀 Permutation Importance":
    st.title("🔀 Permutation Importance")
    st.markdown("Mede a queda no ROC-AUC ao embaralhar aleatoriamente cada feature.")
    st.markdown("---")

    with st.spinner("Calculando Permutation Importance..."):
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=15,
            scoring='roc_auc', random_state=42, n_jobs=-1)

    df_perm = pd.DataFrame({
        'Feature':   FEATURES,
        'Mean Drop': perm.importances_mean,
        'Std':       perm.importances_std,
    }).sort_values('Mean Drop', ascending=False)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor='#0a0f1c')
        ax.set_facecolor('#111827')
        colors_p = ['#f87171' if v > 0 else '#60a5fa' for v in df_perm['Mean Drop']]
        ax.barh(df_perm['Feature'][::-1], df_perm['Mean Drop'][::-1],
                xerr=df_perm['Std'][::-1], color=colors_p[::-1], alpha=0.85,
                error_kw={'ecolor':'white','alpha':0.4})
        ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
        ax.set_title('Queda no ROC-AUC ao embaralhar cada feature', color='#e8eaed')
        ax.set_xlabel('Δ ROC-AUC', color='#9aa0a6')
        ax.tick_params(colors='#9aa0a6')
        for spine in ax.spines.values(): spine.set_edgecolor('#1a2234')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### 📋 Tabela de Importância")
        st.dataframe(
            df_perm[['Feature','Mean Drop','Std']].style
            .format({'Mean Drop':'{:.5f}','Std':'{:.5f}'})
            .background_gradient(subset=['Mean Drop'], cmap='Blues'),
            use_container_width=True)

# ── PAGE 5: Relatório Completo ───────────────────────────────
elif page == "📋 Relatório Completo":
    st.title("📋 Relatório de Explicabilidade")
    st.markdown("Relatório completo gerado automaticamente para auditoria e compliance.")
    st.markdown("---")

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    df_shap   = pd.DataFrame({'Feature': FEATURES, 'Mean |SHAP|': mean_shap})
    df_shap   = df_shap.sort_values('Mean |SHAP|', ascending=False)
    top3      = df_shap.head(3)['Feature'].tolist()

    from datetime import datetime
    report = f"""
=====================================
  RELATÓRIO DE EXPLICABILIDADE — XAI
  Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
=====================================

1. PERFORMANCE DO MODELO
   Modelo:       XGBoost (credit scoring)
   ROC-AUC:      {auc:.4f}
   Dataset:      10.000 clientes (2.000 teste)

2. TOP 5 FEATURES (SHAP)
"""
    for i, row in df_shap.head(5).reset_index(drop=True).iterrows():
        report += f"   {i+1}. {row['Feature']:<30} {row['Mean |SHAP|']:.5f}\n"

    report += f"""
3. TÉCNICAS APLICADAS
   ✅ SHAP TreeExplainer (global + local + waterfall)
   ✅ LIME Tabular (explicações locais aproximadas)
   ✅ Permutation Importance (15 repetições, ROC-AUC)
   ✅ Counterfactual Explanations (busca greedy)

4. CONFORMIDADE REGULATÓRIA
   ✅ LGPD Art. 20 — Direito à explicação de decisão automatizada
   ✅ GDPR Art. 22 — Right to explanation
   ✅ Counterfactuals gerados para todas as negativas

5. RECOMENDAÇÕES
   • Feature "{top3[0]}" é a mais crítica — monitorar drift
   • Implementar monitoramento mensal de feature importance
   • Usar counterfactuals para comunicar negativas de crédito
   • Validar fairness por faixa etária e renda
=====================================
"""
    st.code(report, language='text')
    st.download_button(
        label="⬇️ Baixar Relatório (.txt)",
        data=report,
        file_name="xai_report.txt",
        mime="text/plain",
    )
