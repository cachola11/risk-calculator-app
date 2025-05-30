import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from risk_calculator import RiskMetricsCalculator
import io

def load_dataframe(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except pd.errors.ParserError:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file, sep=';')
        except pd.errors.ParserError:
            uploaded_file.seek(0)
            try:
                return pd.read_excel(uploaded_file)
            except Exception:
                st.error("Não foi possível ler o ficheiro. Verifique o formato e o separador.")
                return None

st.set_page_config(page_title="Calculadora de Risco (VaR e ES)", layout="wide")

# --- Sidebar: Inputs ---
with st.sidebar:
    st.title("⚙️ Parâmetros")
    uploaded_file = st.file_uploader("Carregue um ficheiro CSV ou Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        df = load_dataframe(uploaded_file)
        if df is None:
            st.stop()
        st.write("Primeiras linhas do DataFrame original:")
        st.write(df.head(10))
        cols_validas = [col for col in df.columns if not (col.startswith("Unnamed") or col.strip() == "")]
        date_col = st.selectbox("Coluna de datas", ["Nenhuma"] + cols_validas)
        fund_cols = st.multiselect("Colunas dos Fundos", cols_validas)
        benchmark_col = st.selectbox("Coluna do Benchmark", cols_validas)
        confidence = st.slider("Nível de confiança", 0.80, 0.99, 0.95, 0.01)
        metrics_selected = st.multiselect("Métricas de risco", ["VaR Histórico", "VaR Paramétrico", "ES Histórico", "ES Paramétrico"], default=["VaR Histórico"])
        dist = st.selectbox("Distribuição paramétrica", ["Normal", "Student's t"])
# --- Main: Educational Content ---
st.title("📊 Calculadora de Métricas de Risco (VaR e ES)")
with st.expander("ℹ️ Aplicações práticas na gestão de risco", expanded=True):
    st.markdown("""
**Aplicações do VaR e ES na Gestão de Risco**

**Requisitos Regulamentares**
- **Basileia III/IV:** Os bancos utilizam o VaR e, cada vez mais, o ES para requisitos de capital de risco de mercado.
- **Solvência II:** As seguradoras usam métricas semelhantes para cálculo de capital baseado no risco.
- **UCITS/AIFMD:** Fundos de investimento utilizam o VaR para monitorização e divulgação do risco.

**Gestão Interna de Risco**
- **Definição de Limites de Risco:** As empresas estabelecem limites de VaR/ES para mesas de operações ou carteiras.
- **Desempenho Ajustado ao Risco:** Métricas como o RAROC (Retorno Ajustado ao Risco sobre o Capital) incorporam VaR/ES na avaliação de performance.
- **Alocação de Capital:** O VaR/ES ajuda a determinar quanto capital alocar a diferentes unidades de negócio com base no seu perfil de risco.

**Gestão de Carteiras**
- **Alocação de Ativos:** Otimização de carteiras com base em métricas de risco e não apenas na volatilidade.
- **Análise de Diversificação:** Comparação entre o VaR isolado e o VaR da carteira para avaliar benefícios de diversificação.
- **Testes de Stress:** Utilização de cenários históricos de perdas extremas para complementar as estimativas de VaR/ES.

**Boas Práticas**
- **Utilizar Múltiplas Métricas:** Combinar VaR, ES, testes de stress e análise de cenários.
- **Backtesting:** Testar regularmente as estimativas de VaR face aos resultados reais para validar os modelos.
- **Considerar Vários Horizontes Temporais:** Analisar o risco tanto no curto como no longo prazo.
- **Transparência:** Comunicar claramente as premissas e limitações das métricas de risco.
- **Análise Complementar:** Considerar outros fatores como risco de liquidez, que podem não estar totalmente refletidos no VaR/ES.

**Desafios e Limitações**
- **Risco de Modelo:** Todos os modelos são simplificações da realidade e sujeitos a erro.
- **Dependências nas Caudas:** Abordagens padrão podem subestimar eventos extremos conjuntos.
- **Condições de Mercado Dinâmicas:** As características de risco mudam ao longo do tempo, especialmente em crises.
- **Considerações de Liquidez:** O VaR/ES padrão não considera o risco de liquidez em mercados sob stress.
- **Prociclicidade:** As métricas de risco tendem a ser baixas em períodos calmos e a aumentar acentuadamente em momentos de stress.
    """)

with st.expander("ℹ️ O que são VaR e ES?"):
    st.markdown("""
**Valor em Risco (VaR) e Expected Shortfall (ES) Explicados**

- **VaR:** Mede a perda potencial máxima de uma carteira para um dado nível de confiança e período.
- **ES:** Mede a perda média esperada nos piores cenários, quando a perda ultrapassa o VaR.

*Consulte a explicação detalhada acima para fórmulas e diferenças.*
    """)

st.markdown("---")

# --- Prepare returns dataframe ---
if uploaded_file and fund_cols and benchmark_col:
    df_numeric = df[fund_cols + [benchmark_col]].replace(",", ".", regex=True)
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    df_returns = df_numeric.pct_change().dropna()

# --- Main: Results ---
if uploaded_file and fund_cols and benchmark_col:
    # --- Tabs for Results ---
    tabs = st.tabs(["Resumo", "Detalhe", "Distribuição", "Sensibilidade", "Comparação"])

    # --- Tab 1: Resumo ---
    with tabs[0]:
        st.header("📋 Tabela Resumo de Métricas de Risco")
        confs = [0.90, 0.95, 0.99]
        for col in fund_cols + [benchmark_col]:
            calc = RiskMetricsCalculator(df_returns[col], name=col)
            df_report = calc.generate_risk_report(
                confidence_levels=confs,
                time_horizon=1,
                dist="t" if dist == "Student's t" else "normal"
            )
            st.markdown(f"#### {col}")
            st.dataframe(df_report)

    # --- Tab 2: Detalhe ---
    with tabs[1]:
        st.header("📊 Resultados das Métricas de Risco")
        st.markdown(f"**Nível de confiança:** {confidence*100:.1f}%")
        for col in fund_cols + [benchmark_col]:
            calc = RiskMetricsCalculator(df_returns[col], name=col)
            st.subheader(col)
            metric_cols = st.columns(len(metrics_selected))
            for idx, metric in enumerate(metrics_selected):
                with metric_cols[idx]:
                    if metric == "VaR Histórico":
                        value = calc.calculate_historical_var(confidence)
                        if np.isnan(value):
                            st.warning("VaR Histórico: N/D")
                        else:
                            st.metric("VaR Histórico", f"{value:.4f}")
                            st.caption(f"Prob. perda > {value*100:.2f}% num dia.")
                    elif metric == "VaR Paramétrico":
                        value = calc.calculate_parametric_var(confidence, dist="t" if dist == "Student's t" else "normal")
                        if np.isnan(value):
                            st.warning("VaR Paramétrico: N/D")
                        else:
                            st.metric(f"VaR Paramétrico ({dist})", f"{value:.4f}")
                            st.caption(f"Prob. perda > {value*100:.2f}% num dia.")
                    elif metric == "ES Histórico":
                        value, _ = calc.calculate_historical_es(confidence)
                        if np.isnan(value):
                            st.warning("ES Histórico: N/D")
                        else:
                            st.metric("ES Histórico", f"{value:.4f}")
                            st.caption(f"Média das piores perdas: {value*100:.2f}%")
                    elif metric == "ES Paramétrico":
                        value = calc.calculate_parametric_es(confidence, dist="t" if dist == "Student's t" else "normal")
                        if np.isnan(value):
                            st.warning("ES Paramétrico: N/D")
                        else:
                            st.metric(f"ES Paramétrico ({dist})", f"{value:.4f}")
                            st.caption(f"Média das piores perdas: {value*100:.2f}%")

    # --- Tab 3: Distribuição ---
    with tabs[2]:
        st.header("📈 Distribuição dos Retornos")
        for col in fund_cols + [benchmark_col]:
            calc = RiskMetricsCalculator(df_returns[col], name=col)
            fig, ax = plt.subplots()
            ax.hist(df_returns[col], bins=50, color='skyblue', edgecolor='black')
            if "VaR Histórico" in metrics_selected:
                var_hist = calc.calculate_historical_var(confidence)
                if not np.isnan(var_hist):
                    ax.axvline(-var_hist, color='red', linestyle='--', label=f"VaR Hist. ({var_hist:.4f})")
            if "VaR Paramétrico" in metrics_selected:
                var_param = calc.calculate_parametric_var(confidence, dist="t" if dist == "Student's t" else "normal")
                if not np.isnan(var_param):
                    ax.axvline(-var_param, color='orange', linestyle='--', label=f"VaR Param. ({var_param:.4f})")
            ax.set_title(f"Distribuição dos Retornos - {col}")
            ax.set_xlabel("Retorno diário")
            ax.set_ylabel("Frequência")
            ax.legend()
            st.pyplot(fig)

    # --- Tab 4: Sensibilidade ---
    with tabs[3]:
        st.header("📈 Sensibilidade ao Nível de Confiança e Horizonte Temporal")
        time_horizons = [1, 5, 10, 20]
        for col in fund_cols + [benchmark_col]:
            calc = RiskMetricsCalculator(df_returns[col], name=col)
            st.markdown(f"#### {col}")
            fig1 = calc.plot_confidence_sensitivity(dist="t" if dist == "Student's t" else "normal")
            st.pyplot(fig1)
            fig2 = calc.plot_time_horizon_sensitivity(time_horizons, dist="t" if dist == "Student's t" else "normal")
            st.pyplot(fig2)

    # --- Tab 5: Comparação ---
    with tabs[4]:
        st.header("📊 Comparação de Distribuições")
        col1 = st.selectbox("Primeiro ativo para comparar", fund_cols + [benchmark_col], key="comp1")
        col2 = st.selectbox("Segundo ativo para comparar", fund_cols + [benchmark_col], key="comp2")
        if col1 and col2 and col1 != col2:
            calc1 = RiskMetricsCalculator(df_returns[col1], name=col1)
            calc2 = RiskMetricsCalculator(df_returns[col2], name=col2)
            fig = calc1.plot_comparative_distributions(
                comparison_returns=calc2.returns,
                comparison_name=col2,
                confidence_level=confidence,
                dist="t" if dist == "Student's t" else "normal"
            )
            st.pyplot(fig)

else:
    st.info("Por favor, carregue os dados e selecione as colunas para análise.")

