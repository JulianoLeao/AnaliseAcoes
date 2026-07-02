"""
comparativo_carteiras_yfinance.py

Objetivo:
Comparar duas carteiras de ações brasileiras com R$ 10.000 investidos em cada uma,
divididos igualmente entre os ativos, usando yfinance.

O script calcula:
1) Quanto valeria hoje um investimento feito há 12 meses.
2) Quanto valeria hoje um investimento feito há 3 anos.
3) Detalhamento por ativo.
4) Gráficos comparativos.
5) Exportação para Excel.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python comparativo_carteiras_yfinance.py

Observação:
- Usa preço ajustado quando disponível, que é uma proxy prática para retorno total
  com proventos/desdobramentos ajustados.
- Para cálculo fiscal/oficial, o ideal é usar eventos corporativos e proventos
  diretamente da B3/RI/CEI.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00

# Carteira 1 inferida pelo gráfico em rosca.
CARTEIRA_1 = [
    "PETR4", "ITUB4", "CPFE3", "TOTS3", "EQTL3", "ECOR3", "VALE3",
    "HAPV3", "RADL3", "CCRO3", "BBDC4", "ABEV3", "MGLU3", "BHIA3"
]

# Carteira 2 inferida pelo print da sua carteira atual.
CARTEIRA_2 = [
    "CMIG3", "VALE3", "ITSA4", "BBSE3", "CPLE3", "PETR4", "GOAU4",
    "BBDC4", "WEGE3", "ITUB4", "TAEE11", "KLBN11", "BBAS3", "SAPR4",
    "ISAE3", "CSAN3"
]

# Mapeamentos necessários/úteis.
# CCRO3 mudou para MOTV3. Para histórico, o Yahoo pode aceitar um ou outro conforme a base.
TICKER_MAP = {
    "CCRO3": "MOTV3",
}


def yf_symbol(ticker_b3: str) -> str:
    """Converte ticker B3 para símbolo do Yahoo Finance."""
    ticker = TICKER_MAP.get(ticker_b3, ticker_b3)
    return f"{ticker}.SA"


def pick_price_column(df: pd.DataFrame) -> str:
    """Prioriza Adj Close; se não existir, usa Close."""
    if "Adj Close" in df.columns:
        return "Adj Close"
    if "Close" in df.columns:
        return "Close"
    raise ValueError("Não encontrei coluna de preço: Adj Close ou Close.")


def get_reference_price(series: pd.Series, reference_date: pd.Timestamp) -> float:
    """
    Pega o primeiro preço disponível em ou após a data de referência.
    Isso evita falhas quando a data cai em fim de semana/feriado.
    """
    s = series.dropna().sort_index()
    s = s[s.index >= reference_date]
    if s.empty:
        return math.nan
    return float(s.iloc[0])


def get_current_price(series: pd.Series) -> float:
    """Pega o último preço disponível."""
    s = series.dropna().sort_index()
    if s.empty:
        return math.nan
    return float(s.iloc[-1])


def download_ticker(ticker_b3: str, period: str = "5y") -> pd.Series:
    """
    Baixa série histórica ajustada de um ativo.
    Retorna uma Series com índice de datas e preços ajustados.
    """
    symbol = yf_symbol(ticker_b3)

    df = yf.download(
        symbol,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if df.empty:
        return pd.Series(dtype=float, name=ticker_b3)

    # Em algumas versões, yfinance pode retornar MultiIndex mesmo com um ticker.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Adj Close", symbol) in df.columns:
            series = df[("Adj Close", symbol)]
        elif ("Close", symbol) in df.columns:
            series = df[("Close", symbol)]
        else:
            # fallback simples
            flat = df.copy()
            flat.columns = [c[0] if isinstance(c, tuple) else c for c in flat.columns]
            col = pick_price_column(flat)
            series = flat[col]
    else:
        col = pick_price_column(df)
        series = df[col]

    series.name = ticker_b3
    return series.dropna()


def simulate_portfolio(
    nome_carteira: str,
    ativos: list[str],
    anos_ou_meses: str,
    data_inicio: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    """
    Simula R$ 10.000 divididos igualmente entre os ativos, comprados na data_inicio.
    Usa preço ajustado para aproximar retorno total.
    """
    valor_por_ativo = VALOR_INICIAL / len(ativos)
    linhas = []

    for ativo in ativos:
        serie = download_ticker(ativo, period="5y")

        preco_inicio = get_reference_price(serie, data_inicio)
        preco_atual = get_current_price(serie)

        if math.isnan(preco_inicio) or math.isnan(preco_atual) or preco_inicio <= 0:
            qtd = math.nan
            valor_atual = math.nan
            retorno_pct = math.nan
            ganho = math.nan
        else:
            qtd = valor_por_ativo / preco_inicio
            valor_atual = qtd * preco_atual
            ganho = valor_atual - valor_por_ativo
            retorno_pct = (valor_atual / valor_por_ativo - 1) * 100

        linhas.append({
            "Carteira": nome_carteira,
            "Janela": anos_ou_meses,
            "Ativo": ativo,
            "Ticker usado no Yahoo": yf_symbol(ativo),
            "Valor inicial por ativo (R$)": valor_por_ativo,
            "Preço ajustado inicial (R$)": preco_inicio,
            "Preço ajustado atual (R$)": preco_atual,
            "Quantidade teórica": qtd,
            "Valor atual estimado (R$)": valor_atual,
            "Ganho/Perda estimado (R$)": ganho,
            "Retorno estimado (%)": retorno_pct,
        })

    detalhe = pd.DataFrame(linhas)

    total_atual = detalhe["Valor atual estimado (R$)"].sum(skipna=True)
    retorno_total = (total_atual / VALOR_INICIAL - 1) * 100
    ativos_com_dados = detalhe["Valor atual estimado (R$)"].notna().sum()

    resumo = {
        "Carteira": nome_carteira,
        "Janela": anos_ou_meses,
        "Ativos": len(ativos),
        "Ativos com dados": ativos_com_dados,
        "Investimento inicial (R$)": VALOR_INICIAL,
        "Valor atual estimado (R$)": total_atual,
        "Ganho/Perda estimado (R$)": total_atual - VALOR_INICIAL,
        "Retorno estimado (%)": retorno_total,
    }

    return detalhe, resumo


def make_bar_chart_resumo(resumo: pd.DataFrame, janela: str, output_path: Path) -> None:
    dados = resumo[resumo["Janela"] == janela].copy()

    plt.figure(figsize=(8, 5))
    plt.bar(dados["Carteira"], dados["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL, linestyle="--", linewidth=1)
    plt.ylabel("Valor atual estimado (R$)")
    plt.title(f"R$ 10.000 investidos - {janela}\nDivisão igual entre ativos, preço ajustado")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def make_bar_chart_ativos(detalhe: pd.DataFrame, carteira: str, janela: str, output_path: Path) -> None:
    dados = detalhe[
        (detalhe["Carteira"] == carteira) &
        (detalhe["Janela"] == janela)
    ].copy()

    dados = dados.sort_values("Valor atual estimado (R$)", ascending=False)

    plt.figure(figsize=(11, 6))
    plt.bar(dados["Ativo"], dados["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL / dados["Ativo"].nunique(), linestyle="--", linewidth=1)
    plt.ylabel("Valor atual estimado por ativo (R$)")
    plt.title(f"{carteira} - {janela}\nValor atual por posição")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    hoje = pd.Timestamp.today().normalize()
    data_12m = hoje - pd.DateOffset(months=12)
    data_3a = hoje - pd.DateOffset(years=3)

    outputs = Path("saida_comparativo_yfinance")
    outputs.mkdir(exist_ok=True)

    detalhes = []
    resumos = []

    for nome, ativos in [
        ("Carteira 1", CARTEIRA_1),
        ("Carteira 2", CARTEIRA_2),
    ]:
        detalhe_12m, resumo_12m = simulate_portfolio(nome, ativos, "12 meses", data_12m)
        detalhe_3a, resumo_3a = simulate_portfolio(nome, ativos, "3 anos", data_3a)

        detalhes.extend([detalhe_12m, detalhe_3a])
        resumos.extend([resumo_12m, resumo_3a])

    detalhe_final = pd.concat(detalhes, ignore_index=True)
    resumo_final = pd.DataFrame(resumos)

    # Arredondamentos para visualização/exportação
    detalhe_export = detalhe_final.copy()
    resumo_export = resumo_final.copy()

    for df in [detalhe_export, resumo_export]:
        for col in df.select_dtypes(include="number").columns:
            df[col] = df[col].round(4)

    excel_path = outputs / "comparativo_carteiras_yfinance.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo_export.to_excel(writer, sheet_name="Resumo", index=False)
        detalhe_export.to_excel(writer, sheet_name="Detalhe por ativo", index=False)

    make_bar_chart_resumo(resumo_final, "12 meses", outputs / "comparativo_12_meses.png")
    make_bar_chart_resumo(resumo_final, "3 anos", outputs / "comparativo_3_anos.png")

    for carteira in ["Carteira 1", "Carteira 2"]:
        for janela in ["12 meses", "3 anos"]:
            nome_arquivo = f"{carteira.lower().replace(' ', '_')}_{janela.replace(' ', '_')}.png"
            make_bar_chart_ativos(detalhe_final, carteira, janela, outputs / nome_arquivo)

    print("\nResumo:")
    print(resumo_export.to_string(index=False))

    print(f"\nArquivos gerados na pasta: {outputs.resolve()}")
    print(f"Excel: {excel_path.resolve()}")
    print("\nObservação: confira ativos sem dados no Excel, se houver.")


if __name__ == "__main__":
    main()
