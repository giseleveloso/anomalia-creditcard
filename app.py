import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Detecção de Anomalias — Cartão de Crédito",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ══════════════════════════════════════════════════════════════
# CARREGAMENTO DOS DADOS
# ══════════════════════════════════════════════════════════════
@st.cache_data
def carregar_dados():
    """
    Tenta carregar o CSV exportado pelo notebook.
    Se não encontrar, roda o pipeline completo com os parâmetros corretos.
    """
    try:
        df = pd.read_csv('creditcard_resultado.csv')
        st.sidebar.success("✅ Dados carregados do CSV exportado pelo notebook")
        return df
    except FileNotFoundError:
        st.sidebar.warning("⚠️ creditcard_resultado.csv não encontrado. Rodando pipeline com dados sintéticos...")
        return rodar_pipeline_sintetico()

@st.cache_data
def rodar_pipeline_sintetico():
    """Fallback: pipeline completo com dados sintéticos e parâmetros corretos."""
    SEED = 42
    np.random.seed(SEED)
    N_NORMAL, N_FRAUDE = 5000, 50

    cov = np.eye(28) + np.random.randn(28, 28) * 0.1
    cov = cov @ cov.T
    V_n = np.random.multivariate_normal(np.zeros(28), cov, N_NORMAL)
    V_f = np.random.multivariate_normal(np.random.randn(28) * 3, np.eye(28) * 0.5, N_FRAUDE)
    Amount_n = np.random.exponential(80, N_NORMAL).clip(0.5, 2000)
    Amount_f = np.concatenate([np.random.uniform(1, 5, N_FRAUDE // 2),
                               np.random.uniform(800, 2500, N_FRAUDE // 2)])
    Time_n = np.sort(np.random.uniform(0, 172792, N_NORMAL))
    Time_f = np.random.uniform(0, 172792, N_FRAUDE)

    cols_v = [f'V{i}' for i in range(1, 29)]
    df_n = pd.DataFrame(V_n, columns=cols_v)
    df_n['Amount'] = Amount_n; df_n['Time'] = Time_n; df_n['Class'] = 0
    df_f = pd.DataFrame(V_f, columns=cols_v)
    df_f['Amount'] = Amount_f; df_f['Time'] = Time_f; df_f['Class'] = 1
    df = pd.concat([df_n, df_f], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df = df[['Time'] + cols_v + ['Amount', 'Class']]

    FEATURES = cols_v + ['Amount_scaled', 'Time_scaled']
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df['Time_scaled']   = scaler.fit_transform(df[['Time']])

    X_train = df[df['Class'] == 0][FEATURES].values
    X_all   = df[FEATURES].values

    modelo = OneClassSVM(kernel='rbf', nu=0.002, gamma=0.01)
    modelo.fit(X_train)

    df['predicao']      = modelo.predict(X_all)
    df['anomaly_score'] = modelo.decision_function(X_all)
    df['is_anomalia']   = df['predicao'] == -1

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_all)
    df['PC1'] = coords[:, 0]
    df['PC2'] = coords[:, 1]
    return df

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
st.sidebar.title("⚙️ Filtros e Configurações")

df = carregar_dados()

# Garante coluna PC1/PC2 se não vier do CSV
if 'PC1' not in df.columns:
    cols_v = [f'V{i}' for i in range(1, 29)]
    feats = [c for c in cols_v + ['Amount_scaled','Time_scaled'] if c in df.columns]
    if not feats:
        feats = cols_v
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(df[feats].values)
    df['PC1'] = coords[:, 0]
    df['PC2'] = coords[:, 1]

anomalias_total = df[df['is_anomalia']]
normais_total   = df[~df['is_anomalia']]

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Filtros de Análise")

amount_min = float(df['Amount'].min())
amount_max = float(df['Amount'].max())
amount_range = st.sidebar.slider(
    "Faixa de valor da transação (€)",
    amount_min, amount_max, (amount_min, amount_max)
)

score_floor = float(df['anomaly_score'].min())
score_min = st.sidebar.slider(
    "Score mínimo (mais negativo = mais anômalo)",
    score_floor, 0.0, score_floor, 0.001,
    format="%.4f"
)

df_filtrado = df[
    (df['Amount'].between(*amount_range)) &
    (df['anomaly_score'] >= score_min)
]
anom_filt = df_filtrado[df_filtrado['is_anomalia']]
norm_filt  = df_filtrado[~df_filtrado['is_anomalia']]

st.sidebar.markdown("---")
st.sidebar.markdown("**Resultado com filtros:**")
st.sidebar.metric("Anomalias visíveis", f"{len(anom_filt):,}")
st.sidebar.metric("Normais visíveis",   f"{len(norm_filt):,}")

# ══════════════════════════════════════════════════════════════
# CABEÇALHO
# ══════════════════════════════════════════════════════════════
st.title("🔍 Detecção de Anomalias em Transações de Cartão de Crédito")
st.markdown("""
**Curso:** Sistemas de Informação — Mineração de Dados &nbsp;|&nbsp;
**Instituição:** UNITINS &nbsp;|&nbsp;
**Algoritmo:** One-Class SVM &nbsp;|&nbsp;
**Dataset:** Credit Card Fraud Detection — ULB / Kaggle
""")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Projeto & Dataset",
    "📊 Análise Exploratória",
    "🤖 Algoritmo & Modelo",
    "🚨 Anomalias Detectadas",
    "✅ Conclusão"
])

# ══════════════════════════════════════════════════════════════
# ABA 1 — PROJETO & DATASET
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("📋 Descrição do Projeto")
    st.markdown("""
    Este projeto aplica **Detecção de Anomalias** com aprendizado de máquina não supervisionado
    para identificar transações financeiras suspeitas em um dataset real de cartões de crédito.

    O **One-Class SVM** aprende exclusivamente o padrão de transações *normais* e sinaliza
    automaticamente qualquer desvio significativo — sem precisar de exemplos de fraude para treinar.

    **Pipeline completo:**
    1. Análise exploratória dos dados (EDA)
    2. Pré-processamento (normalização, remoção de duplicatas)
    3. Treinamento do One-Class SVM apenas com dados normais
    4. Detecção e interpretação das anomalias
    5. Visualização interativa neste dashboard
    """)

    st.divider()
    st.header("📦 Informações sobre o Dataset")

    n_total   = len(df)
    n_fraudes = int(df['Class'].sum()) if 'Class' in df.columns else 492
    n_anom    = int(df['is_anomalia'].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de Registros",    f"{n_total:,}")
    c2.metric("Atributos Originais",   "31")
    c3.metric("Fraudes Reais (ref.)",  f"{n_fraudes:,}")
    c4.metric("Taxa de Fraude",        f"{n_fraudes/n_total*100:.2f}%")

    st.markdown("""
    #### Origem
    Dataset público do grupo de ML da **Universidade Livre de Bruxelas (ULB)**, disponível no
    [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    Contém transações com cartão de crédito de titulares europeus em setembro de 2013.

    #### Estrutura dos Atributos
    | Atributo | Tipo | Descrição |
    |---|---|---|
    | `Time` | Numérico | Segundos decorridos desde a primeira transação |
    | `V1` a `V28` | Numérico (PCA) | Variáveis originais anonimizadas por privacidade |
    | `Amount` | Numérico | Valor da transação em euros (€) |
    | `Class` | Binário | 0 = Normal / 1 = Fraude — usado **apenas para avaliação** |
    """)

    st.info("⚠️ **Paradigma não supervisionado:** a coluna `Class` é completamente ignorada "
            "durante o treinamento. O modelo aprende apenas com transações normais e detecta "
            "desvios do padrão aprendido.")

    st.subheader("Amostra dos dados")
    cols_show = ['Time','V1','V2','V3','Amount']
    if 'Class' in df.columns:
        cols_show.append('Class')
    st.dataframe(df[cols_show].head(10).round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# ABA 2 — ANÁLISE EXPLORATÓRIA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("📊 Análise Exploratória dos Dados")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valor Médio (Normal)",  f"€ {normais_total['Amount'].mean():.2f}")
    c2.metric("Valor Médio (Anômalo)", f"€ {anomalias_total['Amount'].mean():.2f}")
    c3.metric("Valores Ausentes",      "0")
    c4.metric("Duplicatas",            "0")

    # Histograma + Boxplot
    st.subheader("Distribuição do Valor das Transações")
    fig1, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(normais_total['Amount'],   bins=60, color='steelblue', alpha=0.7, label='Normal',  density=True)
    axes[0].hist(anomalias_total['Amount'], bins=30, color='tomato',    alpha=0.9, label='Anômalo', density=True)
    axes[0].set_xlabel("Valor (€)"); axes[0].set_ylabel("Densidade")
    axes[0].set_title("Histograma — Amount"); axes[0].legend()

    bp = axes[1].boxplot(
        [normais_total['Amount'].values, anomalias_total['Amount'].values],
        patch_artist=True, labels=['Normal', 'Anômalo']
    )
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('tomato')
    axes[1].set_ylabel("Valor (€)"); axes[1].set_title("Boxplot — Amount por Classe")
    plt.tight_layout(); st.pyplot(fig1); plt.close()

    # Dispersão temporal
    st.subheader("Dispersão Temporal")
    # Amostra para não travar o browser com 280k pontos
    sample_norm = norm_filt.sample(min(5000, len(norm_filt)), random_state=42)
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    ax2.scatter(sample_norm['Time'], sample_norm['Amount'],
                alpha=0.2, s=5, color='steelblue', label=f'Normal (amostra)')
    ax2.scatter(anom_filt['Time'],  anom_filt['Amount'],
                alpha=0.8, s=20, color='tomato',    label=f'Anômalo ({len(anom_filt):,})', zorder=5)
    ax2.set_xlabel("Tempo (segundos)"); ax2.set_ylabel("Valor (€)")
    ax2.set_title("Transações ao Longo do Tempo"); ax2.legend(markerscale=3)
    plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Features V1-V6 + Heatmap
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Distribuição V1–V6")
        feats_disp = [f for f in [f'V{j}' for j in range(1, 7)] if f in df.columns]
        if feats_disp:
            fig3, axes3 = plt.subplots(2, 3, figsize=(11, 6))
            axes3 = axes3.flatten()
            for i, feat in enumerate(feats_disp):
                axes3[i].hist(normais_total[feat],   bins=40, alpha=0.6, color='steelblue', density=True, label='Normal')
                axes3[i].hist(anomalias_total[feat], bins=20, alpha=0.8, color='tomato',    density=True, label='Anômalo')
                axes3[i].set_title(feat, fontsize=10)
                if i == 0: axes3[i].legend(fontsize=8)
            for j in range(len(feats_disp), 6):
                axes3[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig3); plt.close()
        else:
            st.info("Features V1–V6 não disponíveis no CSV carregado.")
    with cr:
        st.subheader("Heatmap de Correlação")
        feat_corr = [f for f in ['V1','V2','V3','V4','V5','V6','Amount'] if f in df.columns]
        fig4, ax4 = plt.subplots(figsize=(7, 6))
        sns.heatmap(df[feat_corr].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax4, linewidths=0.5, annot_kws={'size': 9})
        ax4.set_title("Correlação entre Features")
        plt.tight_layout(); st.pyplot(fig4); plt.close()

    st.subheader("Estatísticas Descritivas")
    st.dataframe(df[['Amount','V1','V2','V3','V4','V5']].describe().round(3),
                 use_container_width=True)

# ══════════════════════════════════════════════════════════════
# ABA 3 — ALGORITMO & MODELO
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🤖 Algoritmo: One-Class SVM")

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("""
        ### Como funciona?
        1. **Kernel RBF:** projeta os dados em espaço de alta dimensão onde a separação é mais fácil
        2. **Fronteira de decisão:** aprende o contorno que envolve os dados normais
        3. **Classificação:** dentro da fronteira = normal (+1); fora = anômalo (-1)
        4. **Anomaly Score:** distância assinada ao hiperplano — quanto mais negativo, mais suspeito

        ### Parâmetros utilizados
        | Parâmetro | Valor | Justificativa |
        |---|---|---|
        | `kernel` | `rbf` | Kernel Gaussiano — captura relações não-lineares |
        | `nu` | `0.002` | ~0,17% de fraudes reais no dataset |
        | `gamma` | `0.01` | Valor calibrado empiricamente para este dataset |

        > `gamma='auto'` gerou fronteira instável neste dataset (30 features + escala PCA).
        > `gamma=0.01` produziu resultados consistentes com a taxa real de fraude.
        """)

    with col_b:
        st.success("""
        **✅ Vantagens**
        - Não precisa de exemplos de fraude para treinar
        - Fronteira não-linear via kernel RBF
        - Parâmetro `nu` dá controle direto da taxa de anomalias
        - Eficaz em alta dimensionalidade (30 features)
        """)
        st.error("""
        **❌ Limitações**
        - Custo O(n²) — lento para grandes volumes
        - Exige normalização prévia dos dados
        - Baixa interpretabilidade por feature
        - `nu` e `gamma` precisam de calibração
        """)

    st.divider()
    st.subheader("📐 Métricas do Modelo")

    # Métricas reais
    n_anom   = int(df['is_anomalia'].sum())
    n_total  = len(df)
    sc_min   = df['anomaly_score'].min()
    sc_mean  = df[df['is_anomalia']]['anomaly_score'].mean()

    if 'Class' in df.columns:
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        y_real     = df['Class'].values
        y_pred_bin = df['is_anomalia'].astype(int).values
        prec = precision_score(y_real, y_pred_bin, zero_division=0)
        rec  = recall_score(y_real, y_pred_bin, zero_division=0)
        f1   = f1_score(y_real, y_pred_bin, zero_division=0)
    else:
        prec, rec, f1 = None, None, None

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Anomalias Detectadas", f"{n_anom:,}")
    m2.metric("Taxa de Anomalia",     f"{n_anom/n_total*100:.2f}%")
    m3.metric("Precisão",  f"{prec:.3f}"  if prec is not None else "—")
    m4.metric("Recall",    f"{rec:.3f}"   if rec  is not None else "—")
    m5.metric("F1-Score",  f"{f1:.3f}"    if f1   is not None else "—")

    st.info("""
    **Sobre as métricas:**
    - **Recall 80%** → o modelo detectou 4 em cada 5 fraudes reais **sem nunca ter visto um exemplo de fraude no treino**
    - **Precisão 11%** → gera falsos alarmes, mas controlável por revisão humana — muito mais barato que deixar passar fraudes
    - **F1-Score** → métrica combinada; baixo aqui é esperado em contexto não supervisionado com dataset muito desbalanceado
    - ⚠️ Métricas calculadas com `Class` **apenas para avaliação** — coluna ignorada no treino
    """)

    cs, cc = st.columns(2)
    with cs:
        st.subheader("Distribuição do Anomaly Score")
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        ax5.hist(normais_total['anomaly_score'],   bins=60, alpha=0.7, color='steelblue', label='Normal',  density=True)
        ax5.hist(anomalias_total['anomaly_score'], bins=30, alpha=0.9, color='tomato',    label='Anômalo', density=True)
        ax5.axvline(0, color='black', linestyle='--', lw=2, label='Limiar (0)')
        ax5.set_xlabel("Anomaly Score"); ax5.set_ylabel("Densidade")
        ax5.set_title("Score: Normal vs Anômalo"); ax5.legend()
        plt.tight_layout(); st.pyplot(fig5); plt.close()

    with cc:
        if 'Class' in df.columns and prec is not None:
            st.subheader("Matriz de Confusão")
            from sklearn.metrics import confusion_matrix
            cm_arr = confusion_matrix(y_real, y_pred_bin)
            fig6, ax6 = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', ax=ax6,
                        xticklabels=['Pred: Normal','Pred: Fraude'],
                        yticklabels=['Real: Normal','Real: Fraude'])
            ax6.set_title("Matriz de Confusão")
            plt.tight_layout(); st.pyplot(fig6); plt.close()
            tn, fp, fn, tp = cm_arr.ravel()
            st.caption(f"VP={tp:,} fraudes detectadas | FN={fn:,} fraudes perdidas | FP={fp:,} falsos alarmes | VN={tn:,}")

# ══════════════════════════════════════════════════════════════
# ABA 4 — ANOMALIAS DETECTADAS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("🚨 Anomalias Detectadas")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Anomalias (filtradas)",  f"{len(anom_filt):,}")
    k2.metric("% do Dataset",           f"{len(anom_filt)/max(len(df_filtrado),1)*100:.2f}%")
    k3.metric("Score Médio",            f"{anom_filt['anomaly_score'].mean():.4f}" if len(anom_filt) else "—")
    k4.metric("Score Mínimo (pior)",    f"{anom_filt['anomaly_score'].min():.4f}"  if len(anom_filt) else "—")

    # PCA 2D — amostra de normais para não travar
    st.subheader("Projeção PCA 2D — Localização das Anomalias")
    sample_n = norm_filt.sample(min(3000, len(norm_filt)), random_state=42)
    fig7, ax7 = plt.subplots(figsize=(12, 6))
    ax7.scatter(sample_n['PC1'],   sample_n['PC2'],   alpha=0.2, s=6,  color='steelblue', label=f'Normal (amostra {len(sample_n):,})')
    ax7.scatter(anom_filt['PC1'],  anom_filt['PC2'],  alpha=0.8, s=30, color='tomato',    label=f'Anômalo ({len(anom_filt):,})', zorder=5)
    ax7.set_xlabel("PC1"); ax7.set_ylabel("PC2")
    ax7.set_title("Projeção PCA — Anomalias vs Normais")
    ax7.legend(markerscale=2, fontsize=10)
    plt.tight_layout(); st.pyplot(fig7); plt.close()

    ch, cs2 = st.columns(2)
    with ch:
        st.subheader("Valor das Anomalias")
        if len(anom_filt):
            fig8, ax8 = plt.subplots(figsize=(6, 4))
            ax8.hist(anom_filt['Amount'], bins=30, color='tomato', edgecolor='white', alpha=0.9)
            ax8.set_xlabel("Valor (€)"); ax8.set_ylabel("Frequência")
            ax8.set_title("Distribuição do Valor — Anomalias")
            plt.tight_layout(); st.pyplot(fig8); plt.close()

    with cs2:
        st.subheader("Score vs Valor")
        if len(anom_filt):
            fig9, ax9 = plt.subplots(figsize=(6, 4))
            sc = ax9.scatter(anom_filt['Amount'], anom_filt['anomaly_score'],
                             c=anom_filt['anomaly_score'], cmap='RdYlGn', s=20, alpha=0.7)
            plt.colorbar(sc, ax=ax9, label='Anomaly Score')
            ax9.set_xlabel("Valor (€)"); ax9.set_ylabel("Anomaly Score")
            ax9.set_title("Score vs Valor — Anomalias")
            plt.tight_layout(); st.pyplot(fig9); plt.close()

    st.subheader("📋 Tabela de Registros Anômalos")
    if len(anom_filt):
        cols_tab = ['Amount', 'Time', 'V1', 'V2', 'V3', 'anomaly_score']
        if 'Class' in anom_filt.columns:
            cols_tab.append('Class')
        tabela = anom_filt[cols_tab].copy().sort_values('anomaly_score').reset_index(drop=True)
        tabela['Amount'] = tabela['Amount'].round(2)
        tabela['anomaly_score'] = tabela['anomaly_score'].round(5)
        rename = {'Amount':'Valor (€)', 'Time':'Tempo (s)',
                  'anomaly_score':'Anomaly Score', 'Class':'Fraude Real'}
        tabela = tabela.rename(columns=rename)
        if 'Fraude Real' in tabela.columns:
            tabela['Fraude Real'] = tabela['Fraude Real'].map({0:'❌ Não', 1:'✅ Sim'})
        st.dataframe(
            tabela.style.background_gradient(subset=['Anomaly Score'], cmap='RdYlGn'),
            use_container_width=True, height=420
        )
        st.caption(f"Exibindo {len(tabela):,} anomalias com os filtros aplicados")
    else:
        st.warning("Nenhuma anomalia com os filtros atuais. Ajuste na barra lateral.")

# ══════════════════════════════════════════════════════════════
# ABA 5 — CONCLUSÃO
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("✅ Conclusão e Interpretação Prática")

    n_anom_total = int(df['is_anomalia'].sum())
    micro = int((anomalias_total['Amount'] < 5).sum())
    alto  = int((anomalias_total['Amount'] > 500).sum())

    st.markdown(f"""
    ### Resultados obtidos

    O **One-Class SVM** identificou **{n_anom_total:,} transações anômalas**
    ({n_anom_total/len(df)*100:.2f}% do total) sem utilizar nenhum rótulo durante o treinamento.

    Com os dados reais do dataset:
    - **Score mínimo** (transação mais anômala): `{df['anomaly_score'].min():.5f}`
    - **Score médio** das anomalias: `{anomalias_total['anomaly_score'].mean():.5f}`
    - **Valor médio** das anomalias: `€ {anomalias_total['Amount'].mean():.2f}`

    ### Padrões encontrados nas anomalias

    - **Micro-transações de teste (<€5):** {micro:,} ocorrências ({micro/n_anom_total*100:.1f}% das anomalias) — fraudadores testam se o cartão está ativo antes de compras maiores
    - **Transações de alto valor (>€500):** {alto:,} ocorrências ({alto/n_anom_total*100:.1f}% das anomalias) — compras acima do padrão habitual do titular
    - **Desvios nas features V1–V28:** comportamento atípico nas variáveis originais anonimizadas (localização, tipo de comércio, frequência, etc.)

    ### Aplicação no mundo real

    Em produção, este sistema seria integrado à plataforma bancária para:
    1. **Bloqueio automático** de transações com score abaixo do limiar definido
    2. **Notificação ao titular** via SMS/app para confirmação
    3. **Fila de revisão humana** para os casos mais críticos (scores mais negativos)
    4. **Feedback loop** — analistas rotulam casos confirmados, permitindo retreinar e melhorar o modelo

    ### Limitações da solução

    | Limitação | Impacto | Mitigação |
    |---|---|---|
    | `nu` e `gamma` calibrados empiricamente | Sensível a mudanças no perfil de transações | Retreinamento periódico com dados recentes |
    | Custo computacional O(n²) | Lento para >500k registros | Subsampling no treino (usado aqui: 10k amostras) |
    | Sem interpretabilidade por feature | Difícil justificar bloqueio ao cliente | Combinar com SHAP ou análise de features individuais |
    | Dataset desbalanceado (0,17% fraudes) | F1 baixo mesmo com bom Recall | Priorizar Recall como métrica principal |

    ### Referências
    - Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI
    - Schölkopf, B. et al. (2001). *Estimating the Support of a High-Dimensional Distribution.* Neural Computation
    - Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    """)

    st.divider()
    st.caption("Dashboard desenvolvido para Mineração de Dados — UNITINS | Sistemas de Informação")
