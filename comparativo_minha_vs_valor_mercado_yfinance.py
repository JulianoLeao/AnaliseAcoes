"""
comparativo_minha_vs_valor_mercado_yfinance.py

Objetivo:
Comparar a sua carteira atual com uma carteira formada pelos maiores ativos por valor de mercado,
conforme print enviado, em duas janelas:
- últimos 12 meses
- últimos 3 anos

Premissa:
- R$ 10.000 investidos em cada carteira.
- Divisão igual entre os ativos.
- Uso de preço ajustado do Yahoo Finance, que funciona como proxy para retorno total
  com dividendos/JCP/desdobramentos ajustados.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python comparativo_minha_vs_valor_mercado_yfinance.py

Saída:
    pasta saida_minha_vs_valor_mercado/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00


# Sua carteira atual, conforme print.
MINHA_CARTEIRA = [
    "CMIG3", "VALE3", "ITSA4", "CPLE3", "BBSE3", "PETR4", "GOAU4",
    "BBDC4", "ITUB4", "WEGE3", "TAEE11", "KLBN11", "BBAS3", "SAPR4",
    "ISAE3", "CSAN3"
]


# Carteira classificada por valor de mercado, conforme print enviado.
# Observação: usei BBDC3 porque é o ticker que aparece no print.
CARTEIRA_VALOR_MERCADO = [
    "PETR4", "ITUB4", "VALE3", "ABEV3", "BPAC11", "WEGE3", "BBDC3",
    "AXIA3", "ITSA4", "BBAS3", "VIVT3", "SANB11", "SBSP3", "B3SA3",
    "RDOR3", "BBSE3"
]


# Ajustes/fallbacks de ticker no Yahoo Finance.
# Em geral o Yahoo usa TICKER.SA para B3.
# Se algum ativo não retornar, inclua aqui um mapeamento alternativo.
TICKER_MAP = {
    # "AXIA3": "AXIA3",  # manter por enquanto; ajuste se o Yahoo não retornar dados.
}


def yf_symbol(ticker_b3: str) -> str:
    ticker = TICKER_MAP.get(ticker_b3, ticker_b3)
    return f"{ticker}.SA"


def get_price_series(symbol: str, start: pd.Timestamp) -> pd.Series:
    """
    Baixa série histórica do Yahoo Finance.
    Prioriza Adj Close; se não existir, usa Close.
    """
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
            # fallback para variações de MultiIndex
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


def build_portfolio_index(nome: str, ativos: list[str], start: pd.Timestamp) -> tuple[pd.Series, pd.DataFrame]:
    """
    Cria um índice de carteira em base 100 com pesos iguais.
    Também retorna tabela de diagnóstico por ativo.
    """
    normalized_assets = []
    diag = []

    for ativo in ativos:
        symbol = yf_symbol(ativo)
        s = get_price_series(symbol, start=start)
        n = normalize_100(s)

        if n.empty:
            diag.append({
                "Carteira": nome,
                "Ativo": ativo,
                "Yahoo": symbol,
                "Status": "Sem dados",
                "Retorno no período (%)": math.nan,
            })
            continue

        n.name = ativo
        normalized_assets.append(n)

        retorno = (float(n.iloc[-1]) / float(n.iloc[0]) - 1) * 100
        diag.append({
            "Carteira": nome,
            "Ativo": ativo,
            "Yahoo": symbol,
            "Status": "OK",
            "Retorno no período (%)": retorno,
        })

    if not normalized_assets:
        return pd.Series(dtype=float, name=nome), pd.DataFrame(diag)

    df_assets = pd.concat(normalized_assets, axis=1).sort_index().ffill()
    portfolio = df_assets.mean(axis=1, skipna=True)
    portfolio.name = nome

    # Rebase para a carteira começar em 100.
    portfolio = normalize_100(portfolio)

    return portfolio, pd.DataFrame(diag)


def simulate_lump_sum_by_asset(nome: str, ativos: list[str], start: pd.Timestamp, janela: str) -> pd.DataFrame:
    """
    Simula R$ 10.000 divididos igualmente entre os ativos,
    medindo quanto cada posição valeria hoje.
    """
    valor_por_ativo = VALOR_INICIAL / len(ativos)
    rows = []

    for ativo in ativos:
        symbol = yf_symbol(ativo)
        s = get_price_series(symbol, start=start)

        if s.empty:
            rows.append({
                "Carteira": nome,
                "Janela": janela,
                "Ativo": ativo,
                "Yahoo": symbol,
                "Valor inicial por ativo (R$)": valor_por_ativo,
                "Preço inicial ajustado": math.nan,
                "Preço atual ajustado": math.nan,
                "Valor atual estimado (R$)": math.nan,
                "Retorno estimado (%)": math.nan,
            })
            continue

        preco_inicial = float(s.iloc[0])
        preco_atual = float(s.iloc[-1])
        valor_atual = valor_por_ativo * (preco_atual / preco_inicial)
        retorno = (valor_atual / valor_por_ativo - 1) * 100

        rows.append({
            "Carteira": nome,
            "Janela": janela,
            "Ativo": ativo,
            "Yahoo": symbol,
            "Valor inicial por ativo (R$)": valor_por_ativo,
            "Preço inicial ajustado": preco_inicial,
            "Preço atual ajustado": preco_atual,
            "Valor atual estimado (R$)": valor_atual,
            "Retorno estimado (%)": retorno,
        })

    return pd.DataFrame(rows)


def plot_time_comparison(df: pd.DataFrame, janela: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))

    for col in df.columns:
        plt.plot(df.index, df[col], linewidth=2, label=col)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"Comparativo no tempo - {janela}\nBase 100, pesos iguais")
    plt.ylabel("Índice base 100")
    plt.xlabel("Data")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_final_bar(resumo: pd.DataFrame, janela: str, output_path: Path) -> None:
    data = resumo[resumo["Janela"] == janela].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(data["Carteira"], data["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL, linestyle="--", linewidth=1)
    plt.title(f"Valor atual de R$ 10.000 - {janela}")
    plt.ylabel("Valor atual estimado (R$)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_asset_bars(detalhe: pd.DataFrame, carteira: str, janela: str, output_path: Path) -> None:
    data = detalhe[
        (detalhe["Carteira"] == carteira) &
        (detalhe["Janela"] == janela)
    ].copy()

    data = data.sort_values("Valor atual estimado (R$)", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(data["Ativo"], data["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL / data["Ativo"].nunique(), linestyle="--", linewidth=1)
    plt.title(f"{carteira} - {janela}\nValor atual por posição")
    plt.ylabel("Valor atual estimado por ativo (R$)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def resumo_from_detalhe(detalhe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (carteira, janela), grupo in detalhe.groupby(["Carteira", "Janela"]):
        valor_atual = grupo["Valor atual estimado (R$)"].sum(skipna=True)
        ativos_com_dados = grupo["Valor atual estimado (R$)"].notna().sum()
        retorno = (valor_atual / VALOR_INICIAL - 1) * 100

        rows.append({
            "Carteira": carteira,
            "Janela": janela,
            "Ativos": len(grupo),
            "Ativos com dados": ativos_com_dados,
            "Investimento inicial (R$)": VALOR_INICIAL,
            "Valor atual estimado (R$)": valor_atual,
            "Ganho/Perda estimado (R$)": valor_atual - VALOR_INICIAL,
            "Retorno estimado (%)": retorno,
        })

    return pd.DataFrame(rows)


def run_window(janela: str, start: pd.Timestamp, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    minha_idx, minha_diag = build_portfolio_index("Minha carteira", MINHA_CARTEIRA, start)
    valor_idx, valor_diag = build_portfolio_index("Carteira valor de mercado", CARTEIRA_VALOR_MERCADO, start)

    df_time = pd.concat([minha_idx, valor_idx], axis=1).dropna().sort_index()
    df_time = df_time / df_time.iloc[0] * 100

    plot_time_comparison(
        df_time,
        janela,
        output_dir / f"comparativo_no_tempo_{janela.replace(' ', '_')}.png",
    )

    detalhe_minha = simulate_lump_sum_by_asset("Minha carteira", MINHA_CARTEIRA, start, janela)
    detalhe_valor = simulate_lump_sum_by_asset("Carteira valor de mercado", CARTEIRA_VALOR_MERCADO, start, janela)
    detalhe = pd.concat([detalhe_minha, detalhe_valor], ignore_index=True)

    diag = pd.concat([minha_diag, valor_diag], ignore_index=True)
    diag["Janela"] = janela

    return df_time, detalhe, diag


def main() -> None:
    hoje = pd.Timestamp.today().normalize()
    start_12m = hoje - pd.DateOffset(months=12)
    start_3a = hoje - pd.DateOffset(years=3)

    output_dir = Path("saida_minha_vs_valor_mercado")
    output_dir.mkdir(exist_ok=True)

    serie_12m, detalhe_12m, diag_12m = run_window("12 meses", start_12m, output_dir)
    serie_3a, detalhe_3a, diag_3a = run_window("3 anos", start_3a, output_dir)

    detalhe = pd.concat([detalhe_12m, detalhe_3a], ignore_index=True)
    diag = pd.concat([diag_12m, diag_3a], ignore_index=True)
    resumo = resumo_from_detalhe(detalhe)

    for janela in ["12 meses", "3 anos"]:
        plot_final_bar(
            resumo,
            janela,
            output_dir / f"comparativo_final_{janela.replace(' ', '_')}.png",
        )

        for carteira in ["Minha carteira", "Carteira valor de mercado"]:
            safe_carteira = carteira.lower().replace(" ", "_")
            safe_janela = janela.replace(" ", "_")
            plot_asset_bars(
                detalhe,
                carteira,
                janela,
                output_dir / f"{safe_carteira}_{safe_janela}_ativos.png",
            )

    excel_path = output_dir / "comparativo_minha_vs_valor_mercado.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo.round(4).to_excel(writer, sheet_name="Resumo", index=False)
        detalhe.round(4).to_excel(writer, sheet_name="Detalhe por ativo", index=False)
        diag.round(4).to_excel(writer, sheet_name="Diagnostico", index=False)
        serie_12m.round(4).to_excel(writer, sheet_name="Serie 12 meses")
        serie_3a.round(4).to_excel(writer, sheet_name="Serie 3 anos")

    print("\nResumo:")
    print(resumo.round(2).to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")

    sem_dados = diag[diag["Status"] != "OK"]
    if not sem_dados.empty:
        print("\nAtenção: ativos sem dados no Yahoo Finance:")
        print(sem_dados.to_string(index=False))


if __name__ == "__main__":
    main()
