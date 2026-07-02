"""
estudo_taxa_real_tesouro_ipca_vs_fiis.py

Objetivo:
Estudar se FIIs fazem sentido como complemento da renda fixa quando as taxas do
Tesouro IPCA+ ficam menos atrativas.

O estudo combina três coisas:

1) Histórico mensal da TAXA OFERECIDA pelo Tesouro IPCA+
   Fonte: Tesouro Transparente / Tesouro Direto
   Dataset: Taxas dos Títulos Ofertados pelo Tesouro Direto

2) IPCA mensal
   Fonte: Banco Central, SGS 433

3) Retorno histórico dos seus blocos:
   - Ações
   - BDRs
   - FIIs sem XPML11
   - Carteiras:
       a) 25/25/25/25
       b) 30/30/15/25
       c) 30/30/40 sem FIIs
       d) 35/35/30 sem FIIs

Perguntas que o estudo ajuda a responder:
- Em quantos meses dos últimos 10 anos havia Tesouro IPCA+ pagando acima de 4,5% real?
- Quando a taxa real do Tesouro cai, os FIIs ficam relativamente mais interessantes?
- FIIs agregaram retorno real acima do IPCA ou só aumentaram risco?
- FIIs melhoram ou pioram a carteira frente a mais Tesouro IPCA+?

Instalação:
    pip install pandas matplotlib openpyxl requests yfinance

Execução:
    python estudo_taxa_real_tesouro_ipca_vs_fiis.py

Saída:
    saida_estudo_taxa_real_tesouro_vs_fiis/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import requests
import yfinance as yf
from io import StringIO
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


VALOR_INICIAL = 10_000.0
TAXA_REAL_ALVO = 4.5

TESOURO_CSV_URL = (
    "https://www.tesourotransparente.gov.br/ckan/dataset/df56aa42-484a-4a59-8184-7676580c81e3/"
    "resource/796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
)


ACOES = [
    "BBSE3", "ITSA4", "CPLE3", "WEGE3", "ITUB4", "CMIG3", "BBDC4", "TAEE11",
    "VALE3", "KLBN11", "PETR4", "BBAS3", "GOAU4", "SAPR4", "ISAE3", "CXSE3"
]

BDRS = [
    "ITLC34", "TSLA34", "TSMC34", "AMZO34", "NVDC34", "AAPL34", "M1TA34",
    "BERK34", "COCA34", "MSFT34"
]

FIIS_SEM_XPML = [
    "GARE11", "VGIR11", "MXRF11", "KNCR11", "VISC11", "XPLG11",
    "VGHF11", "KNRI11", "HGRE11", "HGLG11", "GGRC11"
]


def br_number_to_float(x):
    if pd.isna(x):
        return math.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return math.nan


def read_tesouro_rates() -> pd.DataFrame:
    """
    Lê o CSV oficial de preços e taxas do Tesouro Direto.

    O dataset costuma vir com separador ';' e decimal ','.
    O código tenta ser robusto a pequenas mudanças de nomes de colunas.
    """
    # Em alguns ambientes Windows/rede corporativa o certificado SSL do Tesouro
    # pode falhar por cadeia autoassinada. Por isso baixamos via requests com
    # verify=False. Para um estudo local, isso evita quebrar o script.
    resp = requests.get(TESOURO_CSV_URL, timeout=60, verify=False)
    resp.raise_for_status()
    csv_text = resp.content.decode("latin1")
    df = pd.read_csv(StringIO(csv_text), sep=";", encoding="latin1")

    # Normaliza nomes.
    df.columns = [c.strip() for c in df.columns]

    # Nomes esperados no CSV do Tesouro Transparente.
    # Exemplos comuns:
    # Tipo Titulo, Data Vencimento, Data Base, Taxa Compra Manha, Taxa Venda Manha, PU Compra Manha...
    col_tipo = next((c for c in df.columns if c.lower() in ["tipo titulo", "tipo_titulo", "tipo título"]), None)
    col_venc = next((c for c in df.columns if "vencimento" in c.lower()), None)
    col_data = next((c for c in df.columns if c.lower() in ["data base", "data_base"]), None)

    # Para "taxa oferecida ao investidor", Taxa Compra é a mais alinhada com o que a pessoa física conseguiria travar.
    # Se não existir, usa Taxa Venda como fallback.
    col_taxa = next((c for c in df.columns if "taxa compra" in c.lower()), None)
    if col_taxa is None:
        col_taxa = next((c for c in df.columns if "taxa venda" in c.lower()), None)

    if not all([col_tipo, col_venc, col_data, col_taxa]):
        raise RuntimeError(
            "Não encontrei as colunas esperadas no CSV do Tesouro. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    out = df[[col_tipo, col_venc, col_data, col_taxa]].copy()
    out.columns = ["tipo_titulo", "data_vencimento", "data_base", "taxa_real_oferecida"]

    out["data_base"] = pd.to_datetime(out["data_base"], dayfirst=True, errors="coerce")
    out["data_vencimento"] = pd.to_datetime(out["data_vencimento"], dayfirst=True, errors="coerce")
    out["taxa_real_oferecida"] = out["taxa_real_oferecida"].apply(br_number_to_float)

    out = out.dropna(subset=["data_base", "data_vencimento", "taxa_real_oferecida"])

    # Mantém apenas Tesouro IPCA+ sem juros semestrais.
    # A ideia é representar o título mais parecido com acumulação de longo prazo.
    out = out[out["tipo_titulo"].astype(str).str.strip().eq("Tesouro IPCA+")].copy()

    out["anos_ate_vencimento"] = (out["data_vencimento"] - out["data_base"]).dt.days / 365.25
    out = out[out["anos_ate_vencimento"] > 0].copy()

    return out


def monthly_tesouro_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resume mês a mês:
    - média das taxas IPCA+ disponíveis
    - menor taxa disponível
    - maior taxa disponível
    - taxa do título mais longo disponível
    - quantidade de títulos disponíveis
    """
    df = df.copy()
    df["mes"] = df["data_base"].dt.to_period("M").dt.to_timestamp()

    rows = []
    for mes, g in df.groupby("mes"):
        g = g.sort_values("data_base")

        # última observação do mês para cada vencimento
        last_per_bond = (
            g.sort_values("data_base")
            .groupby("data_vencimento", as_index=False)
            .tail(1)
        )

        longest = last_per_bond.sort_values("data_vencimento").tail(1).iloc[0]
        nearest_long = last_per_bond[last_per_bond["anos_ate_vencimento"] >= 5]
        if not nearest_long.empty:
            nearest_5y = nearest_long.sort_values("anos_ate_vencimento").head(1).iloc[0]
        else:
            nearest_5y = last_per_bond.sort_values("anos_ate_vencimento").tail(1).iloc[0]

        rows.append({
            "mes": mes,
            "taxa_media_ipca_mais": last_per_bond["taxa_real_oferecida"].mean(),
            "taxa_minima_ipca_mais": last_per_bond["taxa_real_oferecida"].min(),
            "taxa_maxima_ipca_mais": last_per_bond["taxa_real_oferecida"].max(),
            "taxa_titulo_mais_longo": float(longest["taxa_real_oferecida"]),
            "vencimento_titulo_mais_longo": longest["data_vencimento"].date(),
            "taxa_titulo_mais_proximo_5a": float(nearest_5y["taxa_real_oferecida"]),
            "vencimento_titulo_mais_proximo_5a": nearest_5y["data_vencimento"].date(),
            "qtd_titulos_ipca_mais": len(last_per_bond),
        })

    return pd.DataFrame(rows).sort_values("mes")


def fetch_ipca_monthly(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
        f"?formato=json&dataInicial={start.strftime('%d/%m/%Y')}&dataFinal={end.strftime('%d/%m/%Y')}"
    )
    r = requests.get(url, timeout=30, verify=False)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        raise RuntimeError("BCB não retornou dados.")
    df = pd.DataFrame(raw)
    df["mes"] = pd.to_datetime(df["data"], dayfirst=True).dt.to_period("M").dt.to_timestamp()
    df["ipca_mensal_pct"] = df["valor"].str.replace(",", ".", regex=False).astype(float)
    return df[["mes", "ipca_mensal_pct"]].sort_values("mes")


def summarize_tesouro_windows(monthly: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    windows = {
        "1 ano": end - pd.DateOffset(years=1),
        "3 anos": end - pd.DateOffset(years=3),
        "5 anos": end - pd.DateOffset(years=5),
        "10 anos": end - pd.DateOffset(years=10),
    }

    rows = []
    for label, start in windows.items():
        g = monthly[(monthly["mes"] >= start.to_period("M").to_timestamp()) & (monthly["mes"] <= end.to_period("M").to_timestamp())].copy()
        if g.empty:
            continue

        col = "taxa_titulo_mais_longo"
        rows.append({
            "Janela": label,
            "Meses analisados": len(g),
            "Taxa real média ofertada - título mais longo (%)": g[col].mean(),
            "Taxa real mínima ofertada - título mais longo (%)": g[col].min(),
            "Taxa real máxima ofertada - título mais longo (%)": g[col].max(),
            "Meses com IPCA+ >= 4,5%": int((g[col] >= TAXA_REAL_ALVO).sum()),
            "% dos meses com IPCA+ >= 4,5%": (g[col] >= TAXA_REAL_ALVO).mean() * 100,
            "Meses com IPCA+ < 4,5%": int((g[col] < TAXA_REAL_ALVO).sum()),
            "% dos meses com IPCA+ < 4,5%": (g[col] < TAXA_REAL_ALVO).mean() * 100,
        })

    return pd.DataFrame(rows)


def yf_symbol(ticker: str) -> str:
    return f"{ticker}.SA"


def get_price_series(symbol: str, start: pd.Timestamp) -> pd.Series:
    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        return pd.Series(dtype=float, name=symbol)

    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", symbol) in df.columns:
            s = df[("Adj Close", symbol)]
        elif ("Close", symbol) in df.columns:
            s = df[("Close", symbol)]
        else:
            flat = df.copy()
            flat.columns = ["_".join([str(x) for x in col if str(x) != ""]).strip() for col in flat.columns]
            adj_cols = [c for c in flat.columns if "Adj Close" in c]
            close_cols = [c for c in flat.columns if "Close" in c and "Adj Close" not in c]
            if adj_cols:
                s = flat[adj_cols[0]]
            elif close_cols:
                s = flat[close_cols[0]]
            else:
                return pd.Series(dtype=float, name=symbol)
    else:
        if "Adj Close" in df.columns:
            s = df["Adj Close"]
        elif "Close" in df.columns:
            s = df["Close"]
        else:
            return pd.Series(dtype=float, name=symbol)

    s = s.dropna().sort_index()
    s.name = symbol
    return s


def normalize_100(series: pd.Series) -> pd.Series:
    s = series.dropna().sort_index()
    if s.empty:
        return s
    first = float(s.iloc[0])
    if first == 0 or math.isnan(first):
        return pd.Series(dtype=float, name=series.name)
    return s / first * 100


def max_drawdown(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return math.nan
    return float((s / s.cummax() - 1).min() * 100)


def annualized_volatility(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 30:
        return math.nan
    ret = s.pct_change().dropna()
    return float(ret.std() * math.sqrt(252) * 100)


def build_asset_class_index(class_name: str, tickers: list[str], start: pd.Timestamp, janela: str):
    normalized = []
    detalhes = []

    for ticker in tickers:
        symbol = yf_symbol(ticker)
        s = get_price_series(symbol, start)

        if s.empty:
            detalhes.append({
                "Janela": janela,
                "Classe": class_name,
                "Ativo": ticker,
                "Status": "Sem dados",
                "Elegível": False,
                "Retorno (%)": math.nan,
            })
            continue

        data_inicial = pd.Timestamp(s.index[0]).normalize()
        dias_apos_inicio = (data_inicial - start).days
        elegivel = dias_apos_inicio <= 10

        status = "OK" if elegivel else f"Histórico parcial: começou {dias_apos_inicio} dias após início"

        if elegivel:
            n = normalize_100(s)
            if not n.empty:
                n.name = ticker
                normalized.append(n)

        ret = (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100

        detalhes.append({
            "Janela": janela,
            "Classe": class_name,
            "Ativo": ticker,
            "Status": status,
            "Elegível": elegivel,
            "Retorno (%)": ret if elegivel else math.nan,
        })

    detalhe = pd.DataFrame(detalhes)

    if not normalized:
        return pd.Series(dtype=float, name=class_name), detalhe

    df = pd.concat(normalized, axis=1, sort=False).sort_index().ffill().dropna()
    serie = df.mean(axis=1, skipna=True)
    serie.name = class_name
    return normalize_100(serie), detalhe


def build_ipca_plus_fixed_series(start: pd.Timestamp, end: pd.Timestamp, ipca: pd.DataFrame, taxa_real_aa: float = 4.5):
    """
    Simula IPCA + taxa_real_aa sem marcação a mercado.
    """
    ipca_use = ipca[(ipca["mes"] >= start.to_period("M").to_timestamp()) & (ipca["mes"] <= end.to_period("M").to_timestamp())].copy()
    real_mensal = (1 + taxa_real_aa / 100) ** (1 / 12) - 1
    ipca_use["retorno"] = (1 + ipca_use["ipca_mensal_pct"] / 100) * (1 + real_mensal) - 1
    ipca_use["indice"] = (1 + ipca_use["retorno"]).cumprod() * 100

    idx = pd.date_range(start=start, end=end, freq="B")
    monthly = pd.Series(ipca_use["indice"].values, index=ipca_use["mes"])
    daily = monthly.reindex(monthly.index.union(idx)).sort_index().ffill().reindex(idx).ffill()
    daily = normalize_100(daily)
    daily.name = f"Tesouro IPCA+{taxa_real_aa:.1f}% travado"
    return daily


def build_weighted_portfolio(series_weights: dict[str, tuple[pd.Series, float]], name: str) -> pd.Series:
    weighted = []
    for label, (serie, peso) in series_weights.items():
        if serie.empty:
            continue
        s = normalize_100(serie) * peso
        s.name = label
        weighted.append(s)

    if not weighted:
        return pd.Series(dtype=float, name=name)

    df = pd.concat(weighted, axis=1, sort=False).sort_index().ffill().dropna()
    p = df.sum(axis=1)
    p.name = name
    return normalize_100(p)


def summarize_series(series: pd.Series, janela: str, item: str, ipca_ret: float):
    s = series.dropna()
    if s.empty:
        return {
            "Janela": janela,
            "Item": item,
            "Retorno nominal acumulado (%)": math.nan,
            "Retorno real acumulado acima do IPCA (%)": math.nan,
            "Volatilidade anualizada (%)": math.nan,
            "Max drawdown (%)": math.nan,
            "Valor final de R$ 10.000": math.nan,
        }

    nominal = float(s.iloc[-1] / s.iloc[0] - 1)
    real = (1 + nominal) / (1 + ipca_ret) - 1
    return {
        "Janela": janela,
        "Item": item,
        "Retorno nominal acumulado (%)": nominal * 100,
        "Retorno real acumulado acima do IPCA (%)": real * 100,
        "Volatilidade anualizada (%)": annualized_volatility(s),
        "Max drawdown (%)": max_drawdown(s),
        "Valor final de R$ 10.000": VALOR_INICIAL * (1 + nominal),
    }


def plot_taxa_real(monthly: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(14, 7))
    plt.plot(monthly["mes"], monthly["taxa_titulo_mais_longo"], linewidth=2, label="Taxa real ofertada - IPCA+ mais longo")
    plt.plot(monthly["mes"], monthly["taxa_media_ipca_mais"], linewidth=1.6, label="Taxa real média dos IPCA+")
    plt.axhline(TAXA_REAL_ALVO, linestyle="--", linewidth=1.5, label="Meta: IPCA+4,5%")
    plt.title("Histórico mensal de taxas reais ofertadas no Tesouro IPCA+")
    plt.xlabel("Mês")
    plt.ylabel("Taxa real ao ano (%)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_portfolios(df: pd.DataFrame, title: str, output_path: Path):
    plt.figure(figsize=(14, 7))
    for col in df.columns:
        plt.plot(df.index, df[col], linewidth=2, label=col)
    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel("Índice base 100")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main():
    hoje = pd.Timestamp.today().normalize()

    output_dir = Path("saida_estudo_taxa_real_tesouro_vs_fiis")
    output_dir.mkdir(exist_ok=True)

    print("Baixando histórico oficial de taxas do Tesouro Direto...")
    tesouro_raw = read_tesouro_rates()
    tesouro_monthly = monthly_tesouro_summary(tesouro_raw)

    # últimos 10 anos para foco do estudo
    start_10y = hoje - pd.DateOffset(years=10)
    tesouro_monthly_10y = tesouro_monthly[tesouro_monthly["mes"] >= start_10y.to_period("M").to_timestamp()].copy()

    print("Baixando IPCA no Banco Central...")
    ipca = fetch_ipca_monthly(start_10y - pd.DateOffset(months=2), hoje)

    # Junta IPCA com taxas do Tesouro para avaliar ambiente mensal.
    monthly_joined = tesouro_monthly_10y.merge(ipca, on="mes", how="left")

    tesouro_windows = summarize_tesouro_windows(tesouro_monthly_10y, hoje)

    plot_taxa_real(tesouro_monthly_10y, output_dir / "taxas_reais_tesouro_ipca_mensal.png")

    janelas = {
        "1 ano": hoje - pd.DateOffset(years=1),
        "3 anos": hoje - pd.DateOffset(years=3),
        "5 anos": hoje - pd.DateOffset(years=5),
        "10 anos": hoje - pd.DateOffset(years=10),
    }

    all_summary = []
    all_details = []

    for janela, start in janelas.items():
        print(f"Processando carteiras - {janela}...")

        ipca_period = ipca[(ipca["mes"] >= start.to_period("M").to_timestamp()) & (ipca["mes"] <= hoje.to_period("M").to_timestamp())].copy()
        ipca_ret = (1 + ipca_period["ipca_mensal_pct"] / 100).prod() - 1

        acoes, det_acoes = build_asset_class_index("Ações", ACOES, start, janela)
        bdrs, det_bdrs = build_asset_class_index("BDRs", BDRS, start, janela)
        fiis, det_fiis = build_asset_class_index("FIIs sem XPML11", FIIS_SEM_XPML, start, janela)
        tesouro_45 = build_ipca_plus_fixed_series(start, hoje, ipca, 4.5)

        carteira_25 = build_weighted_portfolio({
            "Ações": (acoes, 0.25),
            "BDRs": (bdrs, 0.25),
            "FIIs sem XPML11": (fiis, 0.25),
            "Tesouro IPCA+4,5%": (tesouro_45, 0.25),
        }, "Carteira 25/25/25/25")

        carteira_301525 = build_weighted_portfolio({
            "Ações": (acoes, 0.30),
            "BDRs": (bdrs, 0.30),
            "FIIs sem XPML11": (fiis, 0.15),
            "Tesouro IPCA+4,5%": (tesouro_45, 0.25),
        }, "Carteira 30/30/15/25")

        carteira_3040 = build_weighted_portfolio({
            "Ações": (acoes, 0.30),
            "BDRs": (bdrs, 0.30),
            "Tesouro IPCA+4,5%": (tesouro_45, 0.40),
        }, "Carteira 30/30/40 sem FIIs")

        carteira_3530 = build_weighted_portfolio({
            "Ações": (acoes, 0.35),
            "BDRs": (bdrs, 0.35),
            "Tesouro IPCA+4,5%": (tesouro_45, 0.30),
        }, "Carteira 35/35/30 sem FIIs")

        for item, serie in [
            ("Ações", acoes),
            ("BDRs", bdrs),
            ("FIIs sem XPML11", fiis),
            ("Tesouro IPCA+4,5% travado", tesouro_45),
            ("Carteira 25/25/25/25", carteira_25),
            ("Carteira 30/30/15/25", carteira_301525),
            ("Carteira 30/30/40 sem FIIs", carteira_3040),
            ("Carteira 35/35/30 sem FIIs", carteira_3530),
        ]:
            all_summary.append(summarize_series(serie, janela, item, ipca_ret))

        all_details.append(pd.concat([det_acoes, det_bdrs, det_fiis], ignore_index=True))

        df_plot = pd.concat(
            [acoes, bdrs, fiis, tesouro_45, carteira_25, carteira_301525, carteira_3040, carteira_3530],
            axis=1,
            sort=False,
        ).sort_index().ffill().dropna()

        if not df_plot.empty:
            df_plot = df_plot / df_plot.iloc[0] * 100
            plot_portfolios(
                df_plot,
                f"Blocos e carteiras - {janela}",
                output_dir / f"blocos_e_carteiras_{janela.replace(' ', '_')}.png"
            )

    summary = pd.DataFrame(all_summary)
    details = pd.concat(all_details, ignore_index=True)

    excel_path = output_dir / "estudo_taxa_real_tesouro_ipca_vs_fiis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        tesouro_windows.round(4).to_excel(writer, sheet_name="Resumo taxas Tesouro", index=False)
        monthly_joined.round(6).to_excel(writer, sheet_name="Taxas mensais Tesouro", index=False)
        summary.round(4).to_excel(writer, sheet_name="Resumo carteiras", index=False)
        details.round(4).to_excel(writer, sheet_name="Detalhe ativos", index=False)

    print("\nResumo das taxas reais ofertadas pelo Tesouro IPCA+ mais longo:")
    print(tesouro_windows.round(2).to_string(index=False))

    print("\nResumo das carteiras:")
    print(summary.round(2).to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
