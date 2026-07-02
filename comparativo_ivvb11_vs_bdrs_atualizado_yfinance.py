"""
comparativo_ivvb11_vs_bdrs_atualizado_yfinance.py

Comparação:
- IVVB11
- Minha carteira atual de BDRs:
  ITLC34, TSLA34, TSMC34, NVDC34, AMZO34, AAPL34, BERK34, M1TA34, COCA34, MSFT34

Janelas:
- 1 ano
- 3 anos
- 5 anos
- 10 anos

Premissas:
- R$ 10.000 investidos em IVVB11.
- R$ 10.000 investidos na carteira de BDRs.
- Carteira de BDRs com pesos iguais entre os 10 ativos.
- Uso de preço ajustado do Yahoo Finance quando disponível.
- Preço ajustado é uma proxy prática de retorno total, ajustando dividendos/desdobramentos.
- Ativos sem histórico completo próximo da data inicial são marcados como histórico parcial.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python comparativo_ivvb11_vs_bdrs_atualizado_yfinance.py

Saída:
    pasta saida_ivvb11_vs_bdrs_atualizado/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00

BDRS_CARTEIRA = [
    "ITLC34",
    "TSLA34",
    "TSMC34",
    "NVDC34",
    "AMZO34",
    "AAPL34",
    "BERK34",
    "M1TA34",
    "COCA34",
    "MSFT34",
]

ATIVO_REFERENCIA = "IVVB11"


def yf_symbol(ticker_b3: str) -> str:
    return f"{ticker_b3}.SA"


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


def evaluate_single_asset(ticker: str, start: pd.Timestamp, janela: str, valor_inicial: float) -> dict:
    symbol = yf_symbol(ticker)
    s = get_price_series(symbol, start)

    if s.empty:
        return {
            "Janela": janela,
            "Ativo": ticker,
            "Yahoo": symbol,
            "Status": "Sem dados",
            "Elegível": False,
            "Data inicial usada": None,
            "Data final usada": None,
            "Preço inicial ajustado (R$)": math.nan,
            "Preço atual ajustado (R$)": math.nan,
            "Valor inicial (R$)": valor_inicial,
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
        "Ativo": ticker,
        "Yahoo": symbol,
        "Status": status,
        "Elegível": elegivel,
        "Data inicial usada": data_inicial_usada.date(),
        "Data final usada": s.index[-1].date(),
        "Preço inicial ajustado (R$)": preco_inicial,
        "Preço atual ajustado (R$)": preco_atual,
        "Valor inicial (R$)": valor_inicial,
        "Valor atual estimado (R$)": valor_atual if elegivel else math.nan,
        "Retorno estimado (%)": retorno if elegivel else math.nan,
    }


def build_single_asset_index(ticker: str, start: pd.Timestamp, nome: str) -> pd.Series:
    s = get_price_series(yf_symbol(ticker), start)
    if s.empty:
        return pd.Series(dtype=float, name=nome)

    data_inicial_usada = pd.Timestamp(s.index[0]).normalize()
    if (data_inicial_usada - start).days > 10:
        return pd.Series(dtype=float, name=nome)

    n = normalize_100(s)
    n.name = nome
    return n


def build_bdr_portfolio(start: pd.Timestamp, janela: str) -> tuple[pd.Series, pd.DataFrame, dict]:
    detalhes = []
    normalized_assets = []
    valor_por_bdr = VALOR_INICIAL / len(BDRS_CARTEIRA)

    for ticker in BDRS_CARTEIRA:
        row = evaluate_single_asset(ticker, start, janela, valor_por_bdr)
        detalhes.append(row)

        if row["Elegível"]:
            s = get_price_series(row["Yahoo"], start)
            n = normalize_100(s)
            if not n.empty:
                n.name = ticker
                normalized_assets.append(n)

    detalhe = pd.DataFrame(detalhes)

    if not normalized_assets:
        serie = pd.Series(dtype=float, name="Minha carteira de BDRs")
    else:
        df_assets = pd.concat(normalized_assets, axis=1).sort_index().ffill().dropna()
        serie = df_assets.mean(axis=1, skipna=True)
        serie.name = "Minha carteira de BDRs"
        serie = normalize_100(serie)

    valor_atual_total = detalhe["Valor atual estimado (R$)"].sum(skipna=True)
    ativos_elegiveis = int(detalhe["Elegível"].sum())

    resumo = {
        "Janela": janela,
        "Carteira": "Minha carteira de BDRs",
        "Ativos totais": len(BDRS_CARTEIRA),
        "Ativos elegíveis": ativos_elegiveis,
        "Investimento inicial (R$)": VALOR_INICIAL,
        "Valor atual estimado (R$)": valor_atual_total,
        "Ganho/Perda estimado (R$)": valor_atual_total - VALOR_INICIAL,
        "Retorno estimado (%)": (valor_atual_total / VALOR_INICIAL - 1) * 100,
    }

    return serie, detalhe, resumo


def plot_time(df: pd.DataFrame, janela: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    for col in df.columns:
        plt.plot(df.index, df[col], linewidth=2.2, label=col)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"IVVB11 vs minha carteira de BDRs - {janela}\nBase 100, retorno ajustado")
    plt.xlabel("Data")
    plt.ylabel("Índice base 100")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_bar(resumo: pd.DataFrame, output_path: Path) -> None:
    pivot = resumo.pivot(index="Janela", columns="Carteira", values="Retorno estimado (%)")
    ordem = ["1 ano", "3 anos", "5 anos", "10 anos"]
    pivot = pivot.loc[[x for x in ordem if x in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("IVVB11 vs minha carteira de BDRs")
    ax.set_ylabel("Retorno acumulado estimado (%)")
    ax.set_xlabel("Janela")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_bdr_assets(detalhe: pd.DataFrame, janela: str, output_path: Path) -> None:
    data = detalhe[(detalhe["Janela"] == janela) & (detalhe["Elegível"] == True)].copy()
    if data.empty:
        return

    data = data.sort_values("Valor atual estimado (R$)", ascending=False)

    plt.figure(figsize=(11, 6))
    plt.bar(data["Ativo"], data["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL / len(BDRS_CARTEIRA), linestyle="--", linewidth=1)
    plt.title(f"Minha carteira de BDRs - {janela}\nValor atual por posição")
    plt.ylabel("Valor atual estimado (R$)")
    plt.xticks(rotation=45, ha="right")
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

    output_dir = Path("saida_ivvb11_vs_bdrs_atualizado")
    output_dir.mkdir(exist_ok=True)

    todos_resumos = []
    todos_detalhes_bdrs = []
    todos_detalhes_ivvb11 = []
    todas_series = {}

    for janela, start in janelas.items():
        print(f"Processando {janela}...")

        ivvb_eval = evaluate_single_asset(ATIVO_REFERENCIA, start, janela, VALOR_INICIAL)
        todos_detalhes_ivvb11.append(pd.DataFrame([ivvb_eval]))

        resumo_ivvb = {
            "Janela": janela,
            "Carteira": "IVVB11",
            "Ativos totais": 1,
            "Ativos elegíveis": 1 if ivvb_eval["Elegível"] else 0,
            "Investimento inicial (R$)": VALOR_INICIAL,
            "Valor atual estimado (R$)": ivvb_eval["Valor atual estimado (R$)"],
            "Ganho/Perda estimado (R$)": ivvb_eval["Valor atual estimado (R$)"] - VALOR_INICIAL if ivvb_eval["Elegível"] else math.nan,
            "Retorno estimado (%)": ivvb_eval["Retorno estimado (%)"],
        }

        ivvb_series = build_single_asset_index(ATIVO_REFERENCIA, start, "IVVB11")

        bdr_series, detalhe_bdr, resumo_bdr = build_bdr_portfolio(start, janela)

        todos_detalhes_bdrs.append(detalhe_bdr)
        todos_resumos.extend([resumo_ivvb, resumo_bdr])

        df_series = pd.concat([ivvb_series, bdr_series], axis=1).dropna()
        if not df_series.empty:
            df_series = df_series / df_series.iloc[0] * 100
            plot_time(df_series, janela, output_dir / f"comparativo_no_tempo_{janela.replace(' ', '_')}.png")
        todas_series[janela] = df_series

        plot_bdr_assets(detalhe_bdr, janela, output_dir / f"bdrs_por_ativo_{janela.replace(' ', '_')}.png")

    resumo = pd.DataFrame(todos_resumos)
    detalhe_bdrs = pd.concat(todos_detalhes_bdrs, ignore_index=True)
    detalhe_ivvb = pd.concat(todos_detalhes_ivvb11, ignore_index=True)

    plot_bar(resumo, output_dir / "resumo_ivvb11_vs_bdrs.png")

    excel_path = output_dir / "comparativo_ivvb11_vs_bdrs_atualizado.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo.round(4).to_excel(writer, sheet_name="Resumo", index=False)
        detalhe_bdrs.round(4).to_excel(writer, sheet_name="Detalhe BDRs", index=False)
        detalhe_ivvb.round(4).to_excel(writer, sheet_name="Detalhe IVVB11", index=False)

        for janela, serie in todas_series.items():
            sheet = f"Serie {janela}".replace(" ", "_")[:31]
            serie.round(4).to_excel(writer, sheet_name=sheet)

    problemas = detalhe_bdrs[detalhe_bdrs["Status"] != "OK"].copy()

    print("\nResumo:")
    print(resumo.round(2).to_string(index=False))

    if not problemas.empty:
        print("\nAtenção: alguns BDRs tiveram histórico parcial ou sem dados:")
        print(problemas[["Janela", "Ativo", "Yahoo", "Status"]].to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
