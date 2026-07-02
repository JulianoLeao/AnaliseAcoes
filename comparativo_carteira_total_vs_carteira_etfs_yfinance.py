"""
comparativo_carteira_total_vs_carteira_etfs_yfinance.py

Objetivo:
Comparar sua carteira total atual contra uma carteira substituta formada pela SOMA dos ETFs:

- BOVA11  -> renda variável Brasil / ações
- IVVB11  -> renda variável exterior / S&P 500
- XFIX11  -> FIIs / IFIX

Janelas:
- 1 ano
- 3 anos
- 5 anos
- 10 anos

Premissas principais:
- R$ 10.000 investidos na sua carteira total.
- R$ 10.000 investidos na carteira de ETFs.
- Sua carteira total: peso igual por ativo.
- Carteira de ETFs: por padrão, peso igual entre BOVA11, IVVB11 e XFIX11.
  Ou seja: 33,33% em cada ETF.
- Uso de preço ajustado do Yahoo Finance quando disponível.
- Preço ajustado é uma proxy prática de retorno total, incorporando ajustes por dividendos/JCP/desdobramentos.

Atenção:
- Alguns FIIs, BDRs e ETFs podem não ter histórico completo de 10 anos.
- O script marca ativos com histórico parcial ou sem dados.
- XFIX11 pode não ter histórico longo no Yahoo Finance. Se isso ocorrer,
  o script mostra no diagnóstico.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python comparativo_carteira_total_vs_carteira_etfs_yfinance.py

Saída:
    pasta saida_carteira_total_vs_carteira_etfs/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00


# ============================================================
# Sua carteira atual
# ============================================================

ACOES = [
    "BBSE3", "CMIG3", "CPLE3", "ITSA4", "VALE3", "GOAU4", "ITUB4", "WEGE3",
    "BBDC4", "KLBN11", "TAEE11", "PETR4", "BBAS3", "SAPR4", "ISAE3", "CSAN3"
]

FIIS = [
    "GARE11", "XPML11", "VGIR11", "MXRF11", "PVBI11", "KNCR11", "VISC11",
    "XPLG11", "VGHF11", "KNRI11", "HGLG11", "HGRE11"
]

BDRS = [
    "ITLC34", "TSLA34", "TSMC34", "NVDC34", "AMZO34", "AAPL34", "BERK34",
    "M1TA34", "COCA34", "MSFT34"
]

MINHA_CARTEIRA_TOTAL = {
    "Ações": ACOES,
    "FIIs": FIIS,
    "BDRs": BDRS,
}


# ============================================================
# Carteira substituta de ETFs
# ============================================================
# Padrão: pesos iguais entre os três ETFs.
# Se quiser simular 25/25/25 e 25% Tesouro, ou outro peso, edite aqui.

CARTEIRA_ETFS = {
    "BOVA11": 1/3,
    "IVVB11": 1/3,
    "XFIX11": 1/3,
}


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
            flat.columns = [
                "_".join([str(x) for x in col if str(x) != ""]).strip()
                for col in flat.columns
            ]
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


def evaluate_asset(ticker: str, start: pd.Timestamp, janela: str, valor_inicial: float, grupo: str) -> dict:
    symbol = yf_symbol(ticker)
    s = get_price_series(symbol, start)

    if s.empty:
        return {
            "Janela": janela,
            "Grupo": grupo,
            "Ativo": ticker,
            "Yahoo": symbol,
            "Status": "Sem dados",
            "Elegível": False,
            "Data inicial usada": None,
            "Data final usada": None,
            "Valor inicial (R$)": valor_inicial,
            "Preço inicial ajustado (R$)": math.nan,
            "Preço atual ajustado (R$)": math.nan,
            "Valor atual estimado (R$)": math.nan,
            "Retorno estimado (%)": math.nan,
        }

    data_inicial_usada = pd.Timestamp(s.index[0]).normalize()
    dias_apos_inicio = (data_inicial_usada - start).days

    elegivel = dias_apos_inicio <= 10
    status = "OK" if elegivel else f"Histórico parcial: começou {dias_apos_inicio} dias após início"

    preco_inicial = float(s.iloc[0])
    preco_atual = float(s.iloc[-1])
    valor_atual = valor_inicial * (preco_atual / preco_inicial)
    retorno = (valor_atual / valor_inicial - 1) * 100

    return {
        "Janela": janela,
        "Grupo": grupo,
        "Ativo": ticker,
        "Yahoo": symbol,
        "Status": status,
        "Elegível": elegivel,
        "Data inicial usada": data_inicial_usada.date(),
        "Data final usada": s.index[-1].date(),
        "Valor inicial (R$)": valor_inicial,
        "Preço inicial ajustado (R$)": preco_inicial,
        "Preço atual ajustado (R$)": preco_atual,
        "Valor atual estimado (R$)": valor_atual if elegivel else math.nan,
        "Retorno estimado (%)": retorno if elegivel else math.nan,
    }


def build_minha_carteira(start: pd.Timestamp, janela: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Sua carteira total com peso igual por ativo.
    """
    ativos = []
    for grupo, lista in MINHA_CARTEIRA_TOTAL.items():
        for ativo in lista:
            ativos.append((grupo, ativo))

    valor_por_ativo = VALOR_INICIAL / len(ativos)

    detalhes = []
    normalized = []

    for grupo, ativo in ativos:
        row = evaluate_asset(ativo, start, janela, valor_por_ativo, grupo)
        detalhes.append(row)

        if row["Elegível"]:
            s = get_price_series(row["Yahoo"], start)
            n = normalize_100(s)
            if not n.empty:
                n.name = ativo
                normalized.append(n)

    detalhe = pd.DataFrame(detalhes)

    if normalized:
        df = pd.concat(normalized, axis=1).sort_index().ffill().dropna()
        serie = df.mean(axis=1, skipna=True)
        serie.name = "Minha carteira atual"
        serie = normalize_100(serie)
    else:
        serie = pd.Series(dtype=float, name="Minha carteira atual")

    valor_atual = detalhe["Valor atual estimado (R$)"].sum(skipna=True)

    resumo = {
        "Janela": janela,
        "Carteira": "Minha carteira atual",
        "Ativos totais": len(ativos),
        "Ativos elegíveis": int(detalhe["Elegível"].sum()),
        "Investimento inicial (R$)": VALOR_INICIAL,
        "Valor atual estimado (R$)": valor_atual,
        "Ganho/Perda estimado (R$)": valor_atual - VALOR_INICIAL,
        "Retorno estimado (%)": (valor_atual / VALOR_INICIAL - 1) * 100,
    }

    return serie, detalhe, resumo


def build_carteira_etfs(start: pd.Timestamp, janela: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    """
    Carteira substituta composta pela soma dos ETFs, com pesos definidos em CARTEIRA_ETFS.
    """
    detalhes = []
    weighted_series = []

    for ticker, peso in CARTEIRA_ETFS.items():
        valor_no_etf = VALOR_INICIAL * peso
        row = evaluate_asset(ticker, start, janela, valor_no_etf, "ETF")
        row["Peso na carteira de ETFs"] = peso
        detalhes.append(row)

        if row["Elegível"]:
            s = get_price_series(row["Yahoo"], start)
            n = normalize_100(s)
            if not n.empty:
                n.name = ticker
                weighted_series.append(n * peso)

    detalhe = pd.DataFrame(detalhes)

    if weighted_series:
        df = pd.concat(weighted_series, axis=1).sort_index().ffill().dropna()
        serie = df.sum(axis=1, skipna=True)
        serie.name = "Carteira ETFs"
        serie = normalize_100(serie)
    else:
        serie = pd.Series(dtype=float, name="Carteira ETFs")

    valor_atual = detalhe["Valor atual estimado (R$)"].sum(skipna=True)

    resumo = {
        "Janela": janela,
        "Carteira": "Carteira ETFs",
        "Ativos totais": len(CARTEIRA_ETFS),
        "Ativos elegíveis": int(detalhe["Elegível"].sum()),
        "Investimento inicial (R$)": VALOR_INICIAL,
        "Valor atual estimado (R$)": valor_atual,
        "Ganho/Perda estimado (R$)": valor_atual - VALOR_INICIAL,
        "Retorno estimado (%)": (valor_atual / VALOR_INICIAL - 1) * 100,
    }

    return serie, detalhe, resumo


def plot_time(df: pd.DataFrame, janela: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    for col in df.columns:
        plt.plot(df.index, df[col], linewidth=2.4, label=col)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"Minha carteira atual vs carteira de ETFs - {janela}\nBase 100")
    plt.xlabel("Data")
    plt.ylabel("Índice base 100")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_summary_bar(resumo: pd.DataFrame, output_path: Path) -> None:
    ordem = ["1 ano", "3 anos", "5 anos", "10 anos"]
    pivot = resumo.pivot(index="Janela", columns="Carteira", values="Retorno estimado (%)")
    pivot = pivot.loc[[x for x in ordem if x in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Minha carteira atual vs carteira de ETFs")
    ax.set_ylabel("Retorno acumulado estimado (%)")
    ax.set_xlabel("Janela")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_etf_breakdown(detalhe_etfs: pd.DataFrame, janela: str, output_path: Path) -> None:
    data = detalhe_etfs[(detalhe_etfs["Janela"] == janela) & (detalhe_etfs["Elegível"] == True)].copy()

    if data.empty:
        return

    data = data.sort_values("Valor atual estimado (R$)", ascending=False)

    plt.figure(figsize=(8, 5))
    plt.bar(data["Ativo"], data["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL / len(CARTEIRA_ETFS), linestyle="--", linewidth=1)
    plt.title(f"Carteira ETFs - {janela}\nValor atual por ETF")
    plt.ylabel("Valor atual estimado (R$)")
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

    output_dir = Path("saida_carteira_total_vs_carteira_etfs")
    output_dir.mkdir(exist_ok=True)

    todos_resumos = []
    todos_detalhes_minha = []
    todos_detalhes_etfs = []
    series_por_janela = {}

    for janela, start in janelas.items():
        print(f"Processando {janela}...")

        serie_minha, detalhe_minha, resumo_minha = build_minha_carteira(start, janela)
        serie_etfs, detalhe_etfs, resumo_etfs = build_carteira_etfs(start, janela)

        todos_resumos.extend([resumo_minha, resumo_etfs])
        todos_detalhes_minha.append(detalhe_minha)
        todos_detalhes_etfs.append(detalhe_etfs)

        df_series = pd.concat([serie_minha, serie_etfs], axis=1).dropna()
        if not df_series.empty:
            df_series = df_series / df_series.iloc[0] * 100
            plot_time(df_series, janela, output_dir / f"comparativo_no_tempo_{janela.replace(' ', '_')}.png")

        series_por_janela[janela] = df_series

        plot_etf_breakdown(detalhe_etfs, janela, output_dir / f"etfs_por_ativo_{janela.replace(' ', '_')}.png")

    resumo = pd.DataFrame(todos_resumos)
    detalhe_minha = pd.concat(todos_detalhes_minha, ignore_index=True)
    detalhe_etfs = pd.concat(todos_detalhes_etfs, ignore_index=True)

    plot_summary_bar(resumo, output_dir / "resumo_minha_vs_carteira_etfs.png")

    excel_path = output_dir / "comparativo_minha_vs_carteira_etfs.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo.round(4).to_excel(writer, sheet_name="Resumo", index=False)
        detalhe_minha.round(4).to_excel(writer, sheet_name="Detalhe minha carteira", index=False)
        detalhe_etfs.round(4).to_excel(writer, sheet_name="Detalhe carteira ETFs", index=False)

        for janela, serie in series_por_janela.items():
            sheet = f"Serie {janela}".replace(" ", "_")[:31]
            serie.round(4).to_excel(writer, sheet_name=sheet)

    problemas = pd.concat([detalhe_minha, detalhe_etfs], ignore_index=True)
    problemas = problemas[problemas["Status"] != "OK"].copy()

    print("\nResumo:")
    print(resumo.round(2).to_string(index=False))

    print("\nPesos usados na carteira de ETFs:")
    for ticker, peso in CARTEIRA_ETFS.items():
        print(f"{ticker}: {peso:.2%}")

    if not problemas.empty:
        print("\nAtenção: alguns ativos/ETFs tiveram histórico parcial ou sem dados:")
        print(problemas[["Janela", "Grupo", "Ativo", "Yahoo", "Status"]].to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
