"""
simulacao_tesouro_ipca_vs_carteira_30_30_40_sem_fiis.py

Objetivo:
Comparar, para 10, 5, 3 e 1 ano:

1) IPCA acumulado
2) Tesouro IPCA + 4,5% a.a. simulado
3) Ganho real do Tesouro acima da inflação
4) Carteira alternativa sem FIIs:
   - 30% Ações
   - 30% BDRs
   - 40% Tesouro IPCA + 4,5% a.a.

Importante:
- O Tesouro aqui é SIMULADO como IPCA + 4,5% ao ano carregado até o vencimento,
  sem marcação a mercado.
- Isso responde: "quanto teria rendido um título que pagasse IPCA + 4,5%?"
- Não é a mesma coisa que comprar e vender um Tesouro IPCA+ antes do vencimento,
  pois aí haveria marcação a mercado.
- IPCA é buscado na API pública do Banco Central, série SGS 433.
- A carteira usa preços ajustados do Yahoo Finance para ações e BDRs.
- FIIs foram zerados nesta simulação.

Instalação:
    pip install yfinance pandas matplotlib openpyxl requests

Execução:
    python simulacao_tesouro_ipca_vs_carteira_30_30_40_sem_fiis.py

Saída:
    saida_tesouro_ipca_vs_carteira_30_30_40_sem_fiis/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests


VALOR_INICIAL = 10_000.0
TAXA_REAL_ANUAL = 0.045

PESO_ACOES = 0.30
PESO_BDRS = 0.30
PESO_TESOURO = 0.40


# ============================================================
# Carteira conforme imagens
# ============================================================

ACOES = [
    "BBSE3", "ITSA4", "CPLE3", "WEGE3", "ITUB4", "CMIG3", "BBDC4", "TAEE11",
    "VALE3", "KLBN11", "PETR4", "BBAS3", "GOAU4", "SAPR4", "ISAE3", "CXSE3"
]

BDRS = [
    "ITLC34", "TSLA34", "TSMC34", "AMZO34", "NVDC34", "AAPL34", "M1TA34",
    "BERK34", "COCA34", "MSFT34"
]


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
    s = series.dropna().sort_index()
    if s.empty:
        return math.nan
    dd = s / s.cummax() - 1
    return float(dd.min() * 100)


def annualized_volatility(series: pd.Series) -> float:
    s = series.dropna().sort_index()
    if len(s) < 30:
        return math.nan
    ret = s.pct_change().dropna()
    return float(ret.std() * math.sqrt(252) * 100)


def cagr(series: pd.Series, years: float) -> float:
    s = series.dropna().sort_index()
    if s.empty or years <= 0:
        return math.nan
    return float(((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) * 100)


def fetch_ipca_monthly(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
        f"?formato=json&dataInicial={start.strftime('%d/%m/%Y')}&dataFinal={end.strftime('%d/%m/%Y')}"
    )

    r = requests.get(url, timeout=30)
    r.raise_for_status()

    raw = r.json()
    if not raw:
        raise RuntimeError("BCB não retornou dados de IPCA.")

    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["data"], dayfirst=True)
    df["ipca_pct"] = df["valor"].str.replace(",", ".", regex=False).astype(float)
    return df[["date", "ipca_pct"]].sort_values("date")


def build_ipca_and_tesouro_series(start: pd.Timestamp, end: pd.Timestamp):
    """
    Cria:
    - série do IPCA acumulado em base 100
    - série do Tesouro IPCA+4,5 em base 100, sem marcação a mercado
    """
    ipca = fetch_ipca_monthly(start - pd.DateOffset(months=2), end)
    ipca = ipca[(ipca["date"] >= start - pd.DateOffset(months=1)) & (ipca["date"] <= end)].copy()

    real_mensal = (1 + TAXA_REAL_ANUAL) ** (1 / 12) - 1

    ipca["retorno_ipca_mensal"] = ipca["ipca_pct"] / 100
    ipca["retorno_tesouro_mensal"] = (1 + ipca["retorno_ipca_mensal"]) * (1 + real_mensal) - 1

    ipca["indice_ipca"] = (1 + ipca["retorno_ipca_mensal"]).cumprod() * 100
    ipca["indice_tesouro"] = (1 + ipca["retorno_tesouro_mensal"]).cumprod() * 100

    idx = pd.date_range(start=start, end=end, freq="B")

    ipca_monthly = pd.Series(ipca["indice_ipca"].values, index=ipca["date"])
    tesouro_monthly = pd.Series(ipca["indice_tesouro"].values, index=ipca["date"])

    ipca_daily = ipca_monthly.reindex(ipca_monthly.index.union(idx)).sort_index().ffill().reindex(idx).ffill()
    tesouro_daily = tesouro_monthly.reindex(tesouro_monthly.index.union(idx)).sort_index().ffill().reindex(idx).ffill()

    ipca_daily = normalize_100(ipca_daily)
    tesouro_daily = normalize_100(tesouro_daily)

    ipca_daily.name = "IPCA acumulado"
    tesouro_daily.name = "Tesouro IPCA+4,5%"

    return ipca_daily, tesouro_daily, ipca


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
                "Yahoo": symbol,
                "Status": "Sem dados",
                "Elegível": False,
                "Data inicial usada": None,
                "Data final usada": None,
                "Retorno do ativo (%)": math.nan,
            })
            continue

        data_inicial = pd.Timestamp(s.index[0]).normalize()
        dias_apos_inicio = (data_inicial - start).days
        elegivel = dias_apos_inicio <= 10

        if elegivel:
            status = "OK"
            n = normalize_100(s)
            if not n.empty:
                n.name = ticker
                normalized.append(n)
        else:
            status = f"Histórico parcial: começou {dias_apos_inicio} dias após início"

        retorno = (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100

        detalhes.append({
            "Janela": janela,
            "Classe": class_name,
            "Ativo": ticker,
            "Yahoo": symbol,
            "Status": status,
            "Elegível": elegivel,
            "Data inicial usada": data_inicial.date(),
            "Data final usada": s.index[-1].date(),
            "Retorno do ativo (%)": retorno if elegivel else math.nan,
        })

    detalhe = pd.DataFrame(detalhes)

    if not normalized:
        return pd.Series(dtype=float, name=class_name), detalhe

    df = pd.concat(normalized, axis=1, sort=False).sort_index().ffill().dropna()
    serie = df.mean(axis=1, skipna=True)
    serie.name = class_name
    return normalize_100(serie), detalhe


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


def summarize_series(series: pd.Series, janela: str, label: str, years: float) -> dict:
    s = series.dropna().sort_index()

    if s.empty:
        return {
            "Janela": janela,
            "Item": label,
            "Valor inicial (R$)": VALOR_INICIAL,
            "Valor final estimado (R$)": math.nan,
            "Ganho/Perda (R$)": math.nan,
            "Retorno acumulado nominal (%)": math.nan,
            "CAGR nominal (%)": math.nan,
            "Volatilidade anualizada (%)": math.nan,
            "Max drawdown (%)": math.nan,
        }

    valor_final = VALOR_INICIAL * float(s.iloc[-1] / s.iloc[0])
    retorno = (valor_final / VALOR_INICIAL - 1) * 100

    return {
        "Janela": janela,
        "Item": label,
        "Valor inicial (R$)": VALOR_INICIAL,
        "Valor final estimado (R$)": valor_final,
        "Ganho/Perda (R$)": valor_final - VALOR_INICIAL,
        "Retorno acumulado nominal (%)": retorno,
        "CAGR nominal (%)": cagr(s, years),
        "Volatilidade anualizada (%)": annualized_volatility(s),
        "Max drawdown (%)": max_drawdown(s),
    }


def summarize_inflation_tesouro(janela: str, years: float, ipca: pd.Series, tesouro: pd.Series) -> dict:
    ipca_ret = float(ipca.iloc[-1] / ipca.iloc[0] - 1)
    tesouro_ret = float(tesouro.iloc[-1] / tesouro.iloc[0] - 1)

    ganho_real_acum = (1 + tesouro_ret) / (1 + ipca_ret) - 1
    ipca_anual = (1 + ipca_ret) ** (1 / years) - 1
    tesouro_nominal_anual = (1 + tesouro_ret) ** (1 / years) - 1
    real_anual = (1 + tesouro_nominal_anual) / (1 + ipca_anual) - 1

    return {
        "Janela": janela,
        "IPCA acumulado (%)": ipca_ret * 100,
        "IPCA anualizado (%)": ipca_anual * 100,
        "Tesouro IPCA+4,5 acumulado nominal (%)": tesouro_ret * 100,
        "Tesouro IPCA+4,5 nominal anualizado (%)": tesouro_nominal_anual * 100,
        "Ganho real acumulado acima do IPCA (%)": ganho_real_acum * 100,
        "Ganho real anualizado acima do IPCA (%)": real_anual * 100,
    }


def add_real_returns(summary: pd.DataFrame, infl: pd.DataFrame) -> pd.DataFrame:
    out = summary.merge(infl[["Janela", "IPCA acumulado (%)"]], on="Janela", how="left")
    nominal = out["Retorno acumulado nominal (%)"] / 100
    ipca = out["IPCA acumulado (%)"] / 100
    out["Retorno real acumulado acima do IPCA (%)"] = ((1 + nominal) / (1 + ipca) - 1) * 100
    return out


def plot_lines(df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(13, 7))
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


def plot_bar(summary_real: pd.DataFrame, output_path: Path) -> None:
    itens = ["Ações", "BDRs", "Tesouro IPCA+4,5%", "Carteira 30/30/40 sem FIIs"]
    subset = summary_real[summary_real["Item"].isin(itens)].copy()

    pivot = subset.pivot(index="Janela", columns="Item", values="Retorno real acumulado acima do IPCA (%)")
    ordem = ["1 ano", "3 anos", "5 anos", "10 anos"]
    pivot = pivot.loc[[x for x in ordem if x in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(14, 7))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Retorno real acumulado acima do IPCA - Carteira 30/30/40 sem FIIs")
    ax.set_ylabel("Retorno real acumulado (%)")
    ax.set_xlabel("Janela")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    hoje = pd.Timestamp.today().normalize()

    janelas = {
        "1 ano": hoje - pd.DateOffset(years=1),
        "3 anos": hoje - pd.DateOffset(years=3),
        "5 anos": hoje - pd.DateOffset(years=5),
        "10 anos": hoje - pd.DateOffset(years=10),
    }

    output_dir = Path("saida_tesouro_ipca_vs_carteira_30_30_40_sem_fiis")
    output_dir.mkdir(exist_ok=True)

    all_summary = []
    all_inflation = []
    all_details = []
    all_ipca_monthly = []

    for janela, start in janelas.items():
        print(f"Processando {janela}...")
        years = (hoje - start).days / 365.25

        ipca_series, tesouro_series, ipca_monthly = build_ipca_and_tesouro_series(start, hoje)

        acoes, det_acoes = build_asset_class_index("Ações", ACOES, start, janela)
        bdrs, det_bdrs = build_asset_class_index("BDRs", BDRS, start, janela)

        carteira = build_weighted_portfolio({
            "Ações": (acoes, PESO_ACOES),
            "BDRs": (bdrs, PESO_BDRS),
            "Tesouro IPCA+4,5%": (tesouro_series, PESO_TESOURO),
        }, "Carteira 30/30/40 sem FIIs")

        all_inflation.append(summarize_inflation_tesouro(janela, years, ipca_series, tesouro_series))

        for label, serie in [
            ("IPCA acumulado", ipca_series),
            ("Tesouro IPCA+4,5%", tesouro_series),
            ("Ações", acoes),
            ("BDRs", bdrs),
            ("Carteira 30/30/40 sem FIIs", carteira),
        ]:
            all_summary.append(summarize_series(serie, janela, label, years))

        detalhes = pd.concat([det_acoes, det_bdrs], ignore_index=True)
        all_details.append(detalhes)

        ipca_monthly["Janela"] = janela
        all_ipca_monthly.append(ipca_monthly)

        df_plot = pd.concat([ipca_series, tesouro_series, acoes, bdrs, carteira], axis=1, sort=False).sort_index().ffill().dropna()
        if not df_plot.empty:
            df_plot = df_plot / df_plot.iloc[0] * 100
            plot_lines(
                df_plot,
                f"IPCA, Tesouro IPCA+4,5 e carteira 30/30/40 sem FIIs - {janela}",
                output_dir / f"comparativo_{janela.replace(' ', '_')}.png",
            )

    summary = pd.DataFrame(all_summary)
    inflation = pd.DataFrame(all_inflation)
    details = pd.concat(all_details, ignore_index=True)
    ipca_monthly = pd.concat(all_ipca_monthly, ignore_index=True)

    summary_real = add_real_returns(summary, inflation)
    plot_bar(summary_real, output_dir / "retorno_real_acima_ipca.png")

    excel_path = output_dir / "tesouro_ipca_vs_carteira_30_30_40_sem_fiis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        inflation.round(4).to_excel(writer, sheet_name="IPCA x Tesouro", index=False)
        summary_real.round(4).to_excel(writer, sheet_name="Resumo carteira", index=False)
        details.round(4).to_excel(writer, sheet_name="Detalhe ativos", index=False)
        ipca_monthly.round(6).to_excel(writer, sheet_name="IPCA mensal usado", index=False)

    problemas = details[details["Status"] != "OK"].copy()

    print("\nIPCA x Tesouro IPCA+4,5:")
    print(inflation.round(2).to_string(index=False))

    print("\nResumo da carteira:")
    print(summary_real.round(2).to_string(index=False))

    if not problemas.empty:
        print("\nAtenção: alguns ativos tiveram histórico parcial ou sem dados:")
        print(problemas[["Janela", "Classe", "Ativo", "Status"]].to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
