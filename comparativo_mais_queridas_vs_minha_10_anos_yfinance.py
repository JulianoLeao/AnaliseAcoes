"""
comparativo_mais_queridas_vs_minha_10_anos_yfinance.py

Objetivo:
Comparar no período de 10 anos:
- Carteira "Mais Queridas" do Investidor10, conforme print enviado
- Minha carteira atual

Premissas:
- R$ 10.000 investidos em cada carteira.
- Divisão igual entre os ativos.
- Uso de preço ajustado do Yahoo Finance, quando disponível.
- Preço ajustado funciona como proxy prática para retorno total
  com dividendos/JCP/desdobramentos ajustados.

Atenção metodológica:
- Nem todos os ativos têm 10 anos completos de histórico.
- O script marca no diagnóstico os ativos com dados parciais.
- Quando um ativo começou a negociar depois do início da janela,
  ele entra a partir do primeiro dado disponível. Isso pode distorcer
  uma comparação de 10 anos.
- Para uma comparação totalmente "limpa" de 10 anos, você pode excluir
  ativos sem histórico completo ou montar a carteira apenas com ativos
  existentes na data inicial.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python comparativo_mais_queridas_vs_minha_10_anos_yfinance.py

Saída:
    pasta saida_mais_queridas_vs_minha_10_anos/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00


CARTEIRA_MAIS_QUERIDAS = [
    "BBAS3", "PETR4", "BBSE3", "ITSA4", "VALE3", "CMIG4", "CXSE3", "TAEE11",
    "WEGE3", "ISAE4", "ITUB4", "EGIE3", "BBDC3", "CSMG3", "KLBN11", "GOAU4"
]


MINHA_CARTEIRA = [
    "CMIG3", "VALE3", "ITSA4", "CPLE3", "BBSE3", "PETR4", "GOAU4", "BBDC4",
    "ITUB4", "WEGE3", "TAEE11", "KLBN11", "BBAS3", "SAPR4", "ISAE3", "CSAN3"
]


CARTEIRAS = {
    "Mais queridas": CARTEIRA_MAIS_QUERIDAS,
    "Minha carteira": MINHA_CARTEIRA,
}


# Ajustes de ticker, se necessário.
# Se o Yahoo não reconhecer algum ativo, inclua aqui.
TICKER_MAP = {
    # "ISAE4": "ISAE4",
    # "ISAE3": "ISAE3",
}


def yf_symbol(ticker_b3: str) -> str:
    ticker = TICKER_MAP.get(ticker_b3, ticker_b3)
    return f"{ticker}.SA"


def get_price_series(symbol: str, start: pd.Timestamp) -> pd.Series:
    """
    Baixa série de preço do Yahoo Finance.
    Prioriza Adj Close; se não houver, usa Close.
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


def simulate_portfolio(
    nome: str,
    ativos: list[str],
    start: pd.Timestamp,
    modo_historico_completo: bool = False,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simula carteira com pesos iguais.

    modo_historico_completo=False:
        Usa todos os ativos com dados, mesmo que alguns tenham começado após a data inicial.

    modo_historico_completo=True:
        Exclui ativos que não tenham dados próximos da data inicial.
        Isso deixa a comparação de 10 anos mais limpa, mas pode remover ativos relevantes.
    """
    valor_por_ativo = VALOR_INICIAL / len(ativos)
    series_normalizadas = []
    detalhes = []

    for ativo in ativos:
        symbol = yf_symbol(ativo)
        s = get_price_series(symbol, start=start)

        if s.empty:
            detalhes.append({
                "Carteira": nome,
                "Ativo": ativo,
                "Yahoo": symbol,
                "Status": "Sem dados",
                "Incluído no índice": False,
                "Data inicial usada": None,
                "Data final usada": None,
                "Valor inicial por ativo (R$)": valor_por_ativo,
                "Preço inicial ajustado (R$)": math.nan,
                "Preço atual ajustado (R$)": math.nan,
                "Valor atual estimado (R$)": math.nan,
                "Retorno estimado (%)": math.nan,
            })
            continue

        data_inicial_usada = pd.Timestamp(s.index[0]).normalize()
        dias_apos_inicio = (data_inicial_usada - start).days

        historico_parcial = dias_apos_inicio > 10

        if historico_parcial:
            status = f"Dados parciais: começou {dias_apos_inicio} dias após início"
        else:
            status = "OK"

        incluir = True
        if modo_historico_completo and historico_parcial:
            incluir = False

        preco_inicial = float(s.iloc[0])
        preco_atual = float(s.iloc[-1])
        valor_atual = valor_por_ativo * (preco_atual / preco_inicial)
        retorno = (valor_atual / valor_por_ativo - 1) * 100

        if incluir:
            n = normalize_100(s)
            n.name = ativo
            series_normalizadas.append(n)

        detalhes.append({
            "Carteira": nome,
            "Ativo": ativo,
            "Yahoo": symbol,
            "Status": status,
            "Incluído no índice": incluir,
            "Data inicial usada": data_inicial_usada.date(),
            "Data final usada": s.index[-1].date(),
            "Valor inicial por ativo (R$)": valor_por_ativo,
            "Preço inicial ajustado (R$)": preco_inicial,
            "Preço atual ajustado (R$)": preco_atual,
            "Valor atual estimado (R$)": valor_atual if incluir else math.nan,
            "Retorno estimado (%)": retorno if incluir else math.nan,
        })

    if not series_normalizadas:
        carteira = pd.Series(dtype=float, name=nome)
    else:
        df_assets = pd.concat(series_normalizadas, axis=1).sort_index().ffill()
        carteira = df_assets.mean(axis=1, skipna=True)
        carteira.name = nome
        carteira = normalize_100(carteira)

    return carteira, pd.DataFrame(detalhes)


def build_comparison(start: pd.Timestamp, modo_historico_completo: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = []
    detalhes = []

    for nome, ativos in CARTEIRAS.items():
        idx, det = simulate_portfolio(nome, ativos, start, modo_historico_completo=modo_historico_completo)
        if not idx.empty:
            indices.append(idx)
        detalhes.append(det)

    serie = pd.concat(indices, axis=1).sort_index().ffill().dropna(how="all")

    # Rebase pela janela comum, para visualização.
    serie = serie.dropna()
    if not serie.empty:
        serie = serie / serie.iloc[0] * 100

    detalhe = pd.concat(detalhes, ignore_index=True)

    resumo_rows = []
    for carteira, grupo in detalhe.groupby("Carteira"):
        valor_atual = grupo["Valor atual estimado (R$)"].sum(skipna=True)
        ativos_incluidos = int(grupo["Incluído no índice"].sum())
        retorno = (valor_atual / VALOR_INICIAL - 1) * 100

        resumo_rows.append({
            "Carteira": carteira,
            "Ativos totais": len(grupo),
            "Ativos incluídos": ativos_incluidos,
            "Ativos excluídos/sem dados": len(grupo) - ativos_incluidos,
            "Investimento inicial (R$)": VALOR_INICIAL,
            "Valor atual estimado (R$)": valor_atual,
            "Ganho/Perda estimado (R$)": valor_atual - VALOR_INICIAL,
            "Retorno estimado (%)": retorno,
        })

    resumo = pd.DataFrame(resumo_rows).sort_values("Retorno estimado (%)", ascending=False)

    return serie, detalhe, resumo


def plot_time_series(serie: pd.DataFrame, titulo: str, output_path: Path) -> None:
    plt.figure(figsize=(13, 7))

    for col in serie.columns:
        plt.plot(serie.index, serie[col], linewidth=2.2, label=col)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(titulo)
    plt.xlabel("Data")
    plt.ylabel("Índice base 100")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_final_bar(resumo: pd.DataFrame, titulo: str, output_path: Path) -> None:
    data = resumo.sort_values("Valor atual estimado (R$)", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(data["Carteira"], data["Valor atual estimado (R$)"])
    plt.axvline(VALOR_INICIAL, linestyle="--", linewidth=1)
    plt.title(titulo)
    plt.xlabel("Valor atual estimado (R$)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_asset_bars(detalhe: pd.DataFrame, carteira: str, titulo: str, output_path: Path) -> None:
    data = detalhe[
        (detalhe["Carteira"] == carteira) &
        (detalhe["Incluído no índice"] == True)
    ].copy()

    data = data.sort_values("Valor atual estimado (R$)", ascending=False)

    plt.figure(figsize=(12, 6))
    plt.bar(data["Ativo"], data["Valor atual estimado (R$)"])
    plt.axhline(VALOR_INICIAL / 16, linestyle="--", linewidth=1)
    plt.title(titulo)
    plt.ylabel("Valor atual por posição (R$)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    hoje = pd.Timestamp.today().normalize()
    start_10a = hoje - pd.DateOffset(years=10)

    output_dir = Path("saida_mais_queridas_vs_minha_10_anos")
    output_dir.mkdir(exist_ok=True)

    # Modo 1: usa todos os ativos disponíveis, mesmo com histórico parcial.
    serie_parcial, detalhe_parcial, resumo_parcial = build_comparison(
        start_10a,
        modo_historico_completo=False,
    )

    plot_time_series(
        serie_parcial,
        "Mais queridas vs minha carteira - 10 anos\nBase 100, pesos iguais, usando ativos com histórico disponível",
        output_dir / "comparativo_no_tempo_10_anos_historico_disponivel.png",
    )

    plot_final_bar(
        resumo_parcial,
        "Valor atual de R$ 10.000 - 10 anos\nHistórico disponível",
        output_dir / "valor_final_10_anos_historico_disponivel.png",
    )

    for carteira in CARTEIRAS:
        safe = carteira.lower().replace(" ", "_")
        plot_asset_bars(
            detalhe_parcial,
            carteira,
            f"{carteira} - 10 anos\nValor atual por ativo, histórico disponível",
            output_dir / f"{safe}_10_anos_ativos_historico_disponivel.png",
        )

    # Modo 2: comparação mais limpa, excluindo ativos sem 10 anos completos.
    serie_completa, detalhe_completa, resumo_completa = build_comparison(
        start_10a,
        modo_historico_completo=True,
    )

    plot_time_series(
        serie_completa,
        "Mais queridas vs minha carteira - 10 anos\nBase 100, apenas ativos com histórico completo",
        output_dir / "comparativo_no_tempo_10_anos_historico_completo.png",
    )

    plot_final_bar(
        resumo_completa,
        "Valor atual de R$ 10.000 - 10 anos\nApenas ativos com histórico completo",
        output_dir / "valor_final_10_anos_historico_completo.png",
    )

    for carteira in CARTEIRAS:
        safe = carteira.lower().replace(" ", "_")
        plot_asset_bars(
            detalhe_completa,
            carteira,
            f"{carteira} - 10 anos\nValor atual por ativo, histórico completo",
            output_dir / f"{safe}_10_anos_ativos_historico_completo.png",
        )

    excel_path = output_dir / "comparativo_mais_queridas_vs_minha_10_anos.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo_parcial.round(4).to_excel(writer, sheet_name="Resumo historico disp", index=False)
        detalhe_parcial.round(4).to_excel(writer, sheet_name="Detalhe historico disp", index=False)
        serie_parcial.round(4).to_excel(writer, sheet_name="Serie historico disp")

        resumo_completa.round(4).to_excel(writer, sheet_name="Resumo historico compl", index=False)
        detalhe_completa.round(4).to_excel(writer, sheet_name="Detalhe historico compl", index=False)
        serie_completa.round(4).to_excel(writer, sheet_name="Serie historico compl")

    print("\nResumo - histórico disponível:")
    print(resumo_parcial.round(2).to_string(index=False))

    print("\nResumo - apenas ativos com histórico completo:")
    print(resumo_completa.round(2).to_string(index=False))

    problemas = detalhe_parcial[detalhe_parcial["Status"] != "OK"].copy()
    if not problemas.empty:
        print("\nAtenção: ativos com histórico parcial ou sem dados:")
        print(problemas[["Carteira", "Ativo", "Yahoo", "Status"]].to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
