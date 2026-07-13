"""
ESTUDO COMPARATIVO DE CARTEIRAS BRASILEIRAS
===========================================

Compara três carteiras:

1) Top_10
2) Nova_12
3) Todos_16

Metodologia:
- Capital inicial dividido igualmente entre os ativos.
- Buy and hold, sem rebalanceamento periódico.
- Preços ajustados pelo yfinance.
- Janelas solicitadas de 10, 5, 3 e 1 ano.
- Em cada janela, TODAS as carteiras e o benchmark começam na MESMA DATA.
- A data comum é a primeira data em que todos os ativos das três carteiras
  e o benchmark possuem preço válido.

Isso evita comparar uma carteira desde 2017 com outra desde 2021.

Instalação:
    pip install yfinance pandas numpy matplotlib openpyxl

Execução:
    python estudo_carteiras_10_12_16_data_comum.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURAÇÕES
# ============================================================

CAPITAL_INICIAL = 100_000.00
BENCHMARK = "BOVA11.SA"

CARTEIRAS: Dict[str, list[str]] = {
    "Top_10": [
        "BPAC11.SA",
        "WEGE3.SA",
        "ITUB4.SA",
        "CXSE3.SA",
        "TIMS3.SA",
        "ITSA4.SA",
        "PETR4.SA",
        "BBDC4.SA",
        "CMIG4.SA",
        "BBSE3.SA",
    ],
    "Nova_12": [
        "BPAC11.SA",
        "ITUB4.SA",
        "BBAS3.SA",
        "BBSE3.SA",
        "WEGE3.SA",
        "TIMS3.SA",
        "PETR4.SA",
        "CMIG4.SA",
        "CPLE3.SA",
        "ISAE3.SA",
        "SAPR4.SA",
        "VALE3.SA",
    ],
    "Todos_16": [
        "BPAC11.SA",
        "WEGE3.SA",
        "ITUB4.SA",
        "CXSE3.SA",
        "TIMS3.SA",
        "ITSA4.SA",
        "PETR4.SA",
        "BBDC4.SA",
        "CMIG4.SA",
        "BBSE3.SA",
        "ISAE3.SA",
        "CPLE3.SA",
        "BBAS3.SA",
        "SAPR4.SA",
        "TAEE11.SA",
        "KLBN11.SA",
    ],
}

JANELAS_ANOS = [10, 5, 3, 1]
PASTA_SAIDA = Path("resultado_estudo_10_12_16_data_comum")


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def baixar_precos(tickers: Iterable[str], inicio: pd.Timestamp) -> pd.DataFrame:
    tickers = sorted(set(tickers))

    print(f"Baixando {len(tickers)} ativos desde {inicio.date()}...")

    dados = yf.download(
        tickers=tickers,
        start=inicio.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=True,
        actions=False,
        group_by="column",
        threads=True,
    )

    if dados.empty:
        raise RuntimeError("O Yahoo Finance não retornou dados.")

    if isinstance(dados.columns, pd.MultiIndex):
        if "Close" not in dados.columns.get_level_values(0):
            raise RuntimeError("A coluna Close não foi encontrada.")
        precos = dados["Close"].copy()
    else:
        precos = dados[["Close"]].copy()
        precos.columns = tickers[:1]

    if isinstance(precos, pd.Series):
        precos = precos.to_frame()

    precos = precos.sort_index()
    precos = precos.replace([np.inf, -np.inf], np.nan)

    colunas_vazias = [c for c in precos.columns if precos[c].dropna().empty]
    if colunas_vazias:
        print("\nATENÇÃO: sem dados para:")
        for ticker in colunas_vazias:
            print(f"  - {ticker}")

    return precos


def encontrar_inicio_comum(
    precos: pd.DataFrame,
    todos_tickers: list[str],
    inicio_desejado: pd.Timestamp,
) -> pd.Timestamp:
    """
    Encontra a primeira data, a partir do início desejado,
    em que todos os ativos e o benchmark possuem preço válido.
    """
    faltantes = [t for t in todos_tickers if t not in precos.columns]
    if faltantes:
        raise RuntimeError(f"Tickers ausentes no download: {faltantes}")

    recorte = precos.loc[
        precos.index >= inicio_desejado,
        todos_tickers
    ].dropna(how="any")

    if recorte.empty:
        raise RuntimeError(
            "Não existe uma data comum com preço válido para todos os ativos."
        )

    return recorte.index[0]


def montar_carteira_buy_hold(
    precos: pd.DataFrame,
    tickers: list[str],
    inicio_comum: pd.Timestamp,
    capital_inicial: float,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Divide o capital igualmente na data comum e mantém as quantidades.
    """
    recorte = precos.loc[precos.index >= inicio_comum, tickers].copy()
    recorte = recorte.ffill().dropna(how="any")

    if recorte.empty:
        raise RuntimeError("Não há dados suficientes para montar a carteira.")

    precos_iniciais = recorte.iloc[0]
    aporte_por_ativo = capital_inicial / len(tickers)
    quantidades = aporte_por_ativo / precos_iniciais

    valores_individuais = recorte.mul(quantidades, axis=1)
    valor_carteira = valores_individuais.sum(axis=1)
    valor_carteira.name = "Valor da carteira"

    return valor_carteira, valores_individuais


def montar_benchmark(
    precos: pd.DataFrame,
    ticker: str,
    inicio_comum: pd.Timestamp,
    capital_inicial: float,
) -> pd.Series:
    serie = precos.loc[precos.index >= inicio_comum, ticker].copy()
    serie = serie.ffill().dropna()

    if serie.empty:
        raise RuntimeError("Benchmark sem dados suficientes.")

    quantidade = capital_inicial / serie.iloc[0]
    valor = serie * quantidade
    valor.name = ticker.replace(".SA", "")
    return valor


def calcular_metricas(serie: pd.Series, capital_inicial: float) -> dict:
    serie = serie.dropna()

    if len(serie) < 2:
        raise ValueError("Série insuficiente para cálculo de métricas.")

    retorno_total = serie.iloc[-1] / serie.iloc[0] - 1

    dias = (serie.index[-1] - serie.index[0]).days
    anos = dias / 365.25
    cagr = (
        (serie.iloc[-1] / serie.iloc[0]) ** (1 / anos) - 1
        if anos > 0
        else np.nan
    )

    retornos_diarios = serie.pct_change().dropna()
    volatilidade = retornos_diarios.std() * math.sqrt(252)

    maximos = serie.cummax()
    drawdown = serie / maximos - 1
    max_drawdown = drawdown.min()

    sharpe_sem_rf = (
        retornos_diarios.mean() / retornos_diarios.std() * math.sqrt(252)
        if retornos_diarios.std() != 0
        else np.nan
    )

    retorno_sobre_vol = (
        cagr / volatilidade
        if volatilidade and not np.isnan(volatilidade)
        else np.nan
    )

    return {
        "Data inicial": serie.index[0].date(),
        "Data final": serie.index[-1].date(),
        "Anos efetivos": anos,
        "Capital inicial": capital_inicial,
        "Valor final": serie.iloc[-1],
        "Retorno total": retorno_total,
        "CAGR": cagr,
        "Volatilidade anual": volatilidade,
        "Máximo drawdown": max_drawdown,
        "Sharpe sem taxa livre de risco": sharpe_sem_rf,
        "CAGR / Volatilidade": retorno_sobre_vol,
    }


def normalizar_base_100(serie: pd.Series) -> pd.Series:
    serie = serie.dropna()
    return serie / serie.iloc[0] * 100


def formatar_planilha(writer: pd.ExcelWriter, nome_aba: str) -> None:
    ws = writer.book[nome_aba]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions

    for coluna in ws.columns:
        maior = 0
        letra = coluna[0].column_letter
        for celula in coluna:
            valor = "" if celula.value is None else str(celula.value)
            maior = max(maior, len(valor))
        ws.column_dimensions[letra].width = min(maior + 2, 30)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    PASTA_SAIDA.mkdir(exist_ok=True)

    hoje = pd.Timestamp.today().normalize()
    inicio_download = hoje - pd.DateOffset(years=max(JANELAS_ANOS), months=2)

    tickers_carteiras = sorted({
        ticker
        for lista in CARTEIRAS.values()
        for ticker in lista
    })

    tickers_necessarios = sorted(set(tickers_carteiras + [BENCHMARK]))

    precos = baixar_precos(tickers_necessarios, inicio_download)

    resumo: list[dict] = []
    series_exportacao: dict[str, pd.DataFrame] = {}
    composicoes_exportacao: dict[str, pd.DataFrame] = {}

    for anos in JANELAS_ANOS:
        inicio_desejado = hoje - pd.DateOffset(years=anos)

        print(f"\n{'=' * 72}")
        print(f"JANELA SOLICITADA: {anos} ano(s)")
        print(f"{'=' * 72}")

        try:
            inicio_comum = encontrar_inicio_comum(
                precos=precos,
                todos_tickers=tickers_necessarios,
                inicio_desejado=inicio_desejado,
            )
        except Exception as exc:
            print(f"Erro ao encontrar início comum: {exc}")
            continue

        print(f"Data comum efetiva: {inicio_comum.date()}")

        series_janela: dict[str, pd.Series] = {}

        for nome, tickers in CARTEIRAS.items():
            try:
                carteira, valores_individuais = montar_carteira_buy_hold(
                    precos=precos,
                    tickers=tickers,
                    inicio_comum=inicio_comum,
                    capital_inicial=CAPITAL_INICIAL,
                )
            except Exception as exc:
                print(f"{nome}: erro: {exc}")
                continue

            metricas = calcular_metricas(carteira, CAPITAL_INICIAL)
            metricas["Janela solicitada"] = f"{anos} ano(s)"
            metricas["Carteira"] = nome
            metricas["Quantidade de ativos"] = len(tickers)
            resumo.append(metricas)

            series_janela[nome] = normalizar_base_100(carteira)
            composicoes_exportacao[f"{anos}a_{nome}"] = valores_individuais

            print(
                f"{nome}: "
                f"retorno {metricas['Retorno total']:.2%} | "
                f"CAGR {metricas['CAGR']:.2%} | "
                f"vol. {metricas['Volatilidade anual']:.2%} | "
                f"drawdown {metricas['Máximo drawdown']:.2%} | "
                f"CAGR/Vol {metricas['CAGR / Volatilidade']:.2f}"
            )

        try:
            benchmark = montar_benchmark(
                precos=precos,
                ticker=BENCHMARK,
                inicio_comum=inicio_comum,
                capital_inicial=CAPITAL_INICIAL,
            )

            metricas_benchmark = calcular_metricas(benchmark, CAPITAL_INICIAL)
            metricas_benchmark["Janela solicitada"] = f"{anos} ano(s)"
            metricas_benchmark["Carteira"] = "BOVA11"
            metricas_benchmark["Quantidade de ativos"] = 1
            resumo.append(metricas_benchmark)

            series_janela["BOVA11"] = normalizar_base_100(benchmark)

            print(
                f"BOVA11: "
                f"retorno {metricas_benchmark['Retorno total']:.2%} | "
                f"CAGR {metricas_benchmark['CAGR']:.2%} | "
                f"vol. {metricas_benchmark['Volatilidade anual']:.2%} | "
                f"drawdown {metricas_benchmark['Máximo drawdown']:.2%}"
            )
        except Exception as exc:
            print(f"BOVA11: erro: {exc}")

        if series_janela:
            comparacao = pd.concat(series_janela, axis=1).dropna(how="all")
            series_exportacao[f"{anos}_anos"] = comparacao

            plt.figure(figsize=(12, 7))
            for coluna in comparacao.columns:
                plt.plot(comparacao.index, comparacao[coluna], label=coluna)

            plt.title(
                f"Comparação de carteiras — janela solicitada de {anos} ano(s)\n"
                f"Início comum: {inicio_comum.date()}"
            )
            plt.xlabel("Data")
            plt.ylabel("Patrimônio normalizado (base 100)")
            plt.legend()
            plt.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(
                PASTA_SAIDA / f"comparacao_data_comum_{anos}_anos.png",
                dpi=160,
            )
            plt.close()

    if not resumo:
        raise RuntimeError("Nenhuma análise pôde ser concluída.")

    df_resumo = pd.DataFrame(resumo)

    ordem_colunas = [
        "Janela solicitada",
        "Carteira",
        "Quantidade de ativos",
        "Data inicial",
        "Data final",
        "Anos efetivos",
        "Capital inicial",
        "Valor final",
        "Retorno total",
        "CAGR",
        "Volatilidade anual",
        "Máximo drawdown",
        "Sharpe sem taxa livre de risco",
        "CAGR / Volatilidade",
    ]
    df_resumo = df_resumo[ordem_colunas]

    arquivo_excel = PASTA_SAIDA / "estudo_carteiras_10_12_16_data_comum.xlsx"

    with pd.ExcelWriter(arquivo_excel, engine="openpyxl") as writer:
        df_resumo.to_excel(writer, sheet_name="Resumo", index=False)
        formatar_planilha(writer, "Resumo")

        for nome, dataframe in series_exportacao.items():
            aba = f"Series_{nome}"[:31]
            dataframe.to_excel(writer, sheet_name=aba)
            formatar_planilha(writer, aba)

        for nome, dataframe in composicoes_exportacao.items():
            aba = f"Comp_{nome}"[:31]
            dataframe.to_excel(writer, sheet_name=aba)
            formatar_planilha(writer, aba)

        ws = writer.book["Resumo"]
        cabecalhos = {celula.value: celula.column for celula in ws[1]}

        for nome_coluna in [
            "Retorno total",
            "CAGR",
            "Volatilidade anual",
            "Máximo drawdown",
        ]:
            coluna = cabecalhos[nome_coluna]
            for linha in range(2, ws.max_row + 1):
                ws.cell(linha, coluna).number_format = "0.00%"

        for nome_coluna in ["Capital inicial", "Valor final"]:
            coluna = cabecalhos[nome_coluna]
            for linha in range(2, ws.max_row + 1):
                ws.cell(linha, coluna).number_format = 'R$ #,##0.00'

        for nome_coluna in [
            "Anos efetivos",
            "Sharpe sem taxa livre de risco",
            "CAGR / Volatilidade",
        ]:
            coluna = cabecalhos[nome_coluna]
            for linha in range(2, ws.max_row + 1):
                ws.cell(linha, coluna).number_format = "0.00"

    csv_resumo = PASTA_SAIDA / "resumo_metricas_data_comum.csv"
    df_resumo.to_csv(
        csv_resumo,
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    print(f"\nArquivos gerados em: {PASTA_SAIDA.resolve()}")
    print(f"- {arquivo_excel.name}")
    print(f"- {csv_resumo.name}")
    print("- 4 gráficos PNG, todos usando data inicial comum")
    print("\nObservação:")
    print(
        "A janela efetiva pode ser menor que a solicitada porque todas as "
        "carteiras e o benchmark precisam começar na mesma data."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecução cancelada pelo usuário.")
        sys.exit(1)
    except Exception as erro:
        print(f"\nERRO: {erro}")
        sys.exit(1)
