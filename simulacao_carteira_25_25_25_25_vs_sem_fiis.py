"""
simulacao_carteira_25_25_25_25_vs_sem_fiis.py

Simula sua carteira atual com:
- 25% Ações
- 25% BDRs
- 25% FIIs
- 25% Tesouro IPCA + 4,5% a.a.

E compara contra uma alternativa que substitui FIIs por Tesouro:
- 25% Ações
- 25% BDRs
- 50% Tesouro IPCA + 4,5% a.a.

Janelas:
- 1 ano
- 3 anos
- 5 anos

Objetivo:
Testar se o bloco de FIIs compensou frente a Tesouro IPCA+4,5%, olhando:
- retorno acumulado
- CAGR
- volatilidade anualizada
- max drawdown

Instalação:
    pip install yfinance pandas matplotlib openpyxl requests

Execução:
    python simulacao_carteira_25_25_25_25_vs_sem_fiis.py

Saída:
    saida_simulacao_25_25_25_25/
"""

from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import requests


VALOR_INICIAL = 10_000.0
REAL_RATE_ANUAL = 0.045

ACOES = [
    "BBSE3", "ITSA4", "CPLE3", "WEGE3", "ITUB4", "CMIG3", "BBDC4", "TAEE11",
    "VALE3", "KLBN11", "PETR4", "BBAS3", "GOAU4", "SAPR4", "ISAE3", "CXSE3"
]

FIIS = [
    "GARE11", "XPML11", "VGIR11", "MXRF11", "KNCR11", "VISC11", "XPLG11",
    "VGHF11", "KNRI11", "HGRE11", "HGLG11", "GGRC11"
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

    df = pd.concat(normalized, axis=1).sort_index().ffill().dropna()
    serie = df.mean(axis=1, skipna=True)
    serie.name = class_name
    return normalize_100(serie), detalhe


def fetch_ipca_monthly(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
        f"?formato=json&dataInicial={start.strftime('%d/%m/%Y')}&dataFinal={end.strftime('%d/%m/%Y')}"
    )
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        raise RuntimeError("BCB não retornou dados.")
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["data"], dayfirst=True)
    df["ipca_pct"] = df["valor"].str.replace(",", ".", regex=False).astype(float)
    return df[["date", "ipca_pct"]].sort_values("date")


def build_ipca_plus_series(start: pd.Timestamp, end: pd.Timestamp):
    status = "BCB SGS 433"

    try:
        ipca = fetch_ipca_monthly(start - pd.DateOffset(months=2), end)
    except Exception as exc:
        status = f"Fallback manual: {exc}"
        years = max((end - start).days / 365.25, 0.1)
        # Aproximações apenas para o script não quebrar se a API do BCB falhar.
        if years <= 1.2:
            ipca_acum = 0.045
        elif years <= 3.2:
            ipca_acum = 0.15
        else:
            ipca_acum = 0.29
        total_return = (1 + ipca_acum) * ((1 + REAL_RATE_ANUAL) ** years)
        idx = pd.date_range(start=start, end=end, freq="B")
        expo = (idx - idx[0]).days / max((idx[-1] - idx[0]).days, 1)
        s = pd.Series(100 * (total_return ** expo), index=idx, name="Tesouro IPCA+4,5%")
        return s, pd.DataFrame({"date": idx, "ipca_pct": math.nan, "Janela": ""}), status

    ipca = ipca[(ipca["date"] >= start - pd.DateOffset(months=1)) & (ipca["date"] <= end)].copy()
    real_mensal = (1 + REAL_RATE_ANUAL) ** (1 / 12) - 1
    ipca["retorno_mensal"] = (1 + ipca["ipca_pct"] / 100) * (1 + real_mensal) - 1
    ipca["indice"] = (1 + ipca["retorno_mensal"]).cumprod() * 100

    idx = pd.date_range(start=start, end=end, freq="B")
    monthly = pd.Series(ipca["indice"].values, index=ipca["date"])
    daily = monthly.reindex(monthly.index.union(idx)).sort_index().ffill().reindex(idx).ffill()
    daily = normalize_100(daily)
    daily.name = "Tesouro IPCA+4,5%"
    return daily, ipca, status


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

    df = pd.concat(weighted, axis=1).sort_index().ffill().dropna()
    p = df.sum(axis=1)
    p.name = name
    return normalize_100(p)


def summarize_series(series: pd.Series, janela: str, label: str, years: float) -> dict:
    s = series.dropna().sort_index()
    if s.empty:
        return {
            "Janela": janela, "Item": label, "Valor inicial (R$)": VALOR_INICIAL,
            "Valor final estimado (R$)": math.nan, "Ganho/Perda (R$)": math.nan,
            "Retorno acumulado (%)": math.nan, "CAGR (%)": math.nan,
            "Volatilidade anualizada (%)": math.nan, "Max drawdown (%)": math.nan,
        }

    valor_final = VALOR_INICIAL * float(s.iloc[-1] / s.iloc[0])
    retorno = (valor_final / VALOR_INICIAL - 1) * 100

    return {
        "Janela": janela,
        "Item": label,
        "Valor inicial (R$)": VALOR_INICIAL,
        "Valor final estimado (R$)": valor_final,
        "Ganho/Perda (R$)": valor_final - VALOR_INICIAL,
        "Retorno acumulado (%)": retorno,
        "CAGR (%)": cagr(s, years),
        "Volatilidade anualizada (%)": annualized_volatility(s),
        "Max drawdown (%)": max_drawdown(s),
    }


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


def plot_summary_bar(summary: pd.DataFrame, output_path: Path) -> None:
    subset = summary[summary["Item"].isin([
        "Carteira 25/25/25/25",
        "Alternativa sem FIIs",
        "FIIs",
        "Tesouro IPCA+4,5%",
    ])].copy()

    pivot = subset.pivot(index="Janela", columns="Item", values="Retorno acumulado (%)")
    ordem = ["1 ano", "3 anos", "5 anos"]
    pivot = pivot.loc[[x for x in ordem if x in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(13, 7))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("FIIs em teste: retorno vs Tesouro e impacto na carteira")
    ax.set_ylabel("Retorno acumulado (%)")
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
    }

    output_dir = Path("saida_simulacao_25_25_25_25")
    output_dir.mkdir(exist_ok=True)

    all_summary = []
    all_detalhes = []
    all_ipca = []

    for janela, start in janelas.items():
        print(f"Processando {janela}...")
        years = (hoje - start).days / 365.25

        acoes, det_acoes = build_asset_class_index("Ações", ACOES, start, janela)
        bdrs, det_bdrs = build_asset_class_index("BDRs", BDRS, start, janela)
        fiis, det_fiis = build_asset_class_index("FIIs", FIIS, start, janela)
        tesouro, ipca_df, ipca_status = build_ipca_plus_series(start, hoje)

        det_rf = pd.DataFrame([{
            "Janela": janela,
            "Classe": "Tesouro",
            "Ativo": "Tesouro IPCA+4,5%",
            "Yahoo": "-",
            "Status": ipca_status,
            "Elegível": True,
            "Data inicial usada": tesouro.index[0].date() if not tesouro.empty else None,
            "Data final usada": tesouro.index[-1].date() if not tesouro.empty else None,
            "Retorno do ativo (%)": (tesouro.iloc[-1] / tesouro.iloc[0] - 1) * 100 if not tesouro.empty else math.nan,
        }])

        all_detalhes.append(pd.concat([det_acoes, det_bdrs, det_fiis, det_rf], ignore_index=True))
        ipca_df["Janela"] = janela
        all_ipca.append(ipca_df)

        carteira_25252525 = build_weighted_portfolio({
            "Ações": (acoes, 0.25),
            "BDRs": (bdrs, 0.25),
            "FIIs": (fiis, 0.25),
            "Tesouro IPCA+4,5%": (tesouro, 0.25),
        }, "Carteira 25/25/25/25")

        alternativa_sem_fiis = build_weighted_portfolio({
            "Ações": (acoes, 0.25),
            "BDRs": (bdrs, 0.25),
            "Tesouro IPCA+4,5%": (tesouro, 0.50),
        }, "Alternativa sem FIIs")

        blocos = pd.concat([acoes, bdrs, fiis, tesouro], axis=1).dropna()
        if not blocos.empty:
            blocos = blocos / blocos.iloc[0] * 100
            plot_lines(
                blocos,
                f"Blocos da carteira - {janela}\nAções, BDRs, FIIs e Tesouro IPCA+4,5",
                output_dir / f"blocos_{janela.replace(' ', '_')}.png",
            )

        carteiras = pd.concat([carteira_25252525, alternativa_sem_fiis], axis=1).dropna()
        if not carteiras.empty:
            carteiras = carteiras / carteiras.iloc[0] * 100
            plot_lines(
                carteiras,
                f"Carteira 25/25/25/25 vs alternativa sem FIIs - {janela}",
                output_dir / f"carteiras_{janela.replace(' ', '_')}.png",
            )

        for label, serie in [
            ("Ações", acoes),
            ("BDRs", bdrs),
            ("FIIs", fiis),
            ("Tesouro IPCA+4,5%", tesouro),
            ("Carteira 25/25/25/25", carteira_25252525),
            ("Alternativa sem FIIs", alternativa_sem_fiis),
        ]:
            all_summary.append(summarize_series(serie, janela, label, years))

    summary = pd.DataFrame(all_summary)
    detalhes_final = pd.concat(all_detalhes, ignore_index=True)
    ipca_final = pd.concat(all_ipca, ignore_index=True)

    plot_summary_bar(summary, output_dir / "resumo_retornos.png")

    excel_path = output_dir / "simulacao_25_25_25_25_vs_sem_fiis.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary.round(4).to_excel(writer, sheet_name="Resumo", index=False)
        detalhes_final.round(4).to_excel(writer, sheet_name="Detalhe ativos", index=False)
        ipca_final.round(6).to_excel(writer, sheet_name="IPCA usado", index=False)

    problemas = detalhes_final[detalhes_final["Status"] != "OK"].copy()

    print("\nResumo:")
    print(summary.round(2).to_string(index=False))

    if not problemas.empty:
        print("\nAtenção: alguns ativos tiveram histórico parcial ou sem dados:")
        print(problemas[["Janela", "Classe", "Ativo", "Status"]].to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
