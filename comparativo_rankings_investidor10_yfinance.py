"""
comparativo_rankings_investidor10_yfinance.py

Objetivo:
Comparar carteiras formadas pelos rankings dos prints enviados, usando o nome do cabeçalho:

1) Ranking de Ações que Nunca Tiveram Prejuízo
2) Ranking de Ações de Maiores Receitas
3) Ranking de Ações Mais Queridas
4) Ranking de Ações de Maiores Lucros
5) Minha carteira atual, opcional, para referência

Janelas:
- 5 anos
- 3 anos
- 1 ano

Premissas:
- R$ 10.000 investidos em cada carteira.
- Divisão igual entre os ativos de cada carteira.
- Uso de preço ajustado do Yahoo Finance, quando disponível.
- O preço ajustado é uma proxy prática para retorno total com dividendos/JCP/desdobramentos.
- Ativos sem dados suficientes são marcados no diagnóstico.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python comparativo_rankings_investidor10_yfinance.py

Saída:
    pasta saida_rankings_investidor10/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00

INCLUIR_MINHA_CARTEIRA = True


CARTEIRAS = {
    "Nunca tiveram prejuízo": [
        "ITSA4", "LOGG3", "BRBI11", "AFLT3", "VTRU3", "B3SA3", "MULT3", "EZTC3",
        "TAEE11", "BNBR3", "ALOS3", "ALUP11", "BPAC11", "GRND3", "ISAE4", "SAUD3"
    ],

    "Maiores receitas": [
        "PETR4", "ITUB4", "BBAS3", "BBDC3", "VALE3", "VBBR3", "SANB11", "MBRF3",
        "UGPA3", "ABEV3", "ASAI3", "GGBR3", "GOAU4", "BPAC11", "BRKM5", "VIVT3"
    ],

    "Mais queridas": [
        "BBAS3", "PETR4", "BBSE3", "ITSA4", "VALE3", "CMIG4", "CXSE3", "TAEE11",
        "WEGE3", "ISAE4", "ITUB4", "EGIE3", "BBDC3", "CSMG3", "KLBN11", "GOAU4"
    ],

    "Maiores lucros": [
        "PETR4", "ITUB4", "BBDC3", "EQPA3", "BPAC11", "ITSA4", "ABEV3", "BBAS3",
        "VALE3", "SANB11", "SUZB3", "AXIA3", "BBSE3", "SBSP3", "WEGE3", "VIVT3"
    ],
}


MINHA_CARTEIRA = [
    "CMIG3", "VALE3", "ITSA4", "CPLE3", "BBSE3", "PETR4", "GOAU4", "BBDC4",
    "ITUB4", "WEGE3", "TAEE11", "KLBN11", "BBAS3", "SAPR4", "ISAE3", "CSAN3"
]

if INCLUIR_MINHA_CARTEIRA:
    CARTEIRAS["Minha carteira atual"] = MINHA_CARTEIRA


# Ajuste de tickers quando necessário.
# Deixe vazio por padrão. Se o Yahoo não reconhecer algum ticker, coloque o ajuste aqui.
TICKER_MAP = {
    # Exemplos:
    # "AXIA3": "AXIA3",
    # "ISAE4": "ISAE4",
}


def yf_symbol(ticker_b3: str) -> str:
    ticker = TICKER_MAP.get(ticker_b3, ticker_b3)
    return f"{ticker}.SA"


def get_price_series(symbol: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.Series:
    """
    Baixa preços históricos do Yahoo Finance.
    Prioriza Adj Close; se não houver, usa Close.
    """
    df = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=None if end is None else end.strftime("%Y-%m-%d"),
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


def simulate_portfolio(nome: str, ativos: list[str], start: pd.Timestamp, janela: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simula R$ 10.000 dividido igualmente entre os ativos.
    Para o gráfico no tempo, cada ativo é normalizado e a carteira é a média simples.
    Para a tabela, calcula valor atual por ativo.
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
                "Janela": janela,
                "Ativo": ativo,
                "Yahoo": symbol,
                "Status": "Sem dados",
                "Data inicial usada": None,
                "Data final usada": None,
                "Valor inicial por ativo (R$)": valor_por_ativo,
                "Preço inicial ajustado (R$)": math.nan,
                "Preço atual ajustado (R$)": math.nan,
                "Valor atual estimado (R$)": math.nan,
                "Ganho/Perda estimado (R$)": math.nan,
                "Retorno estimado (%)": math.nan,
            })
            continue

        n = normalize_100(s)
        n.name = ativo
        series_normalizadas.append(n)

        preco_inicial = float(s.iloc[0])
        preco_atual = float(s.iloc[-1])
        valor_atual = valor_por_ativo * (preco_atual / preco_inicial)
        ganho = valor_atual - valor_por_ativo
        retorno = (valor_atual / valor_por_ativo - 1) * 100

        data_inicial_usada = s.index[0].date()
        data_final_usada = s.index[-1].date()

        # Se o ativo não tem dados próximos ao início da janela, marcar como parcial.
        dias_apos_inicio = (pd.Timestamp(s.index[0]).normalize() - start).days
        if dias_apos_inicio > 10:
            status = f"Dados parciais: começou {dias_apos_inicio} dias após início"
        else:
            status = "OK"

        detalhes.append({
            "Carteira": nome,
            "Janela": janela,
            "Ativo": ativo,
            "Yahoo": symbol,
            "Status": status,
            "Data inicial usada": data_inicial_usada,
            "Data final usada": data_final_usada,
            "Valor inicial por ativo (R$)": valor_por_ativo,
            "Preço inicial ajustado (R$)": preco_inicial,
            "Preço atual ajustado (R$)": preco_atual,
            "Valor atual estimado (R$)": valor_atual,
            "Ganho/Perda estimado (R$)": ganho,
            "Retorno estimado (%)": retorno,
        })

    if not series_normalizadas:
        carteira = pd.Series(dtype=float, name=nome)
    else:
        df_assets = pd.concat(series_normalizadas, axis=1).sort_index().ffill()
        carteira = df_assets.mean(axis=1, skipna=True)
        carteira.name = nome
        carteira = normalize_100(carteira)

    return carteira, pd.DataFrame(detalhes)


def build_window(janela: str, start: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = []
    detalhes = []

    for nome, ativos in CARTEIRAS.items():
        idx, det = simulate_portfolio(nome, ativos, start, janela)
        if not idx.empty:
            indices.append(idx)
        detalhes.append(det)

    serie = pd.concat(indices, axis=1).sort_index().ffill().dropna(how="all")
    # Rebase para período comum. Isso facilita comparação visual.
    serie = serie.dropna()
    if not serie.empty:
        serie = serie / serie.iloc[0] * 100

    detalhe = pd.concat(detalhes, ignore_index=True)

    resumo_rows = []
    for (carteira, janela_nome), grupo in detalhe.groupby(["Carteira", "Janela"]):
        valor_atual = grupo["Valor atual estimado (R$)"].sum(skipna=True)
        ativos = len(grupo)
        ativos_ok = grupo["Valor atual estimado (R$)"].notna().sum()

        resumo_rows.append({
            "Carteira": carteira,
            "Janela": janela_nome,
            "Ativos": ativos,
            "Ativos com dados": ativos_ok,
            "Investimento inicial (R$)": VALOR_INICIAL,
            "Valor atual estimado (R$)": valor_atual,
            "Ganho/Perda estimado (R$)": valor_atual - VALOR_INICIAL,
            "Retorno estimado (%)": (valor_atual / VALOR_INICIAL - 1) * 100,
        })

    resumo = pd.DataFrame(resumo_rows).sort_values(
        ["Janela", "Retorno estimado (%)"],
        ascending=[True, False],
    )

    return serie, detalhe, resumo


def plot_time_series(serie: pd.DataFrame, janela: str, output_path: Path) -> None:
    plt.figure(figsize=(14, 7))

    for col in serie.columns:
        plt.plot(serie.index, serie[col], linewidth=2, label=col)

    plt.axhline(100, linestyle="--", linewidth=1)
    plt.title(f"Comparativo dos rankings - {janela}\nBase 100, pesos iguais")
    plt.xlabel("Data")
    plt.ylabel("Índice base 100")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_final_values(resumo: pd.DataFrame, janela: str, output_path: Path) -> None:
    data = resumo[resumo["Janela"] == janela].copy()
    data = data.sort_values("Valor atual estimado (R$)", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(data["Carteira"], data["Valor atual estimado (R$)"])
    plt.axvline(VALOR_INICIAL, linestyle="--", linewidth=1)
    plt.title(f"Valor atual de R$ 10.000 - {janela}")
    plt.xlabel("Valor atual estimado (R$)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_returns(resumo: pd.DataFrame, janela: str, output_path: Path) -> None:
    data = resumo[resumo["Janela"] == janela].copy()
    data = data.sort_values("Retorno estimado (%)", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(data["Carteira"], data["Retorno estimado (%)"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(f"Retorno estimado por carteira - {janela}")
    plt.xlabel("Retorno estimado (%)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    hoje = pd.Timestamp.today().normalize()

    janelas = {
        "5 anos": hoje - pd.DateOffset(years=5),
        "3 anos": hoje - pd.DateOffset(years=3),
        "1 ano": hoje - pd.DateOffset(years=1),
    }

    output_dir = Path("saida_rankings_investidor10")
    output_dir.mkdir(exist_ok=True)

    todas_series = {}
    todos_detalhes = []
    todos_resumos = []

    for janela, start in janelas.items():
        serie, detalhe, resumo = build_window(janela, start)

        todas_series[janela] = serie
        todos_detalhes.append(detalhe)
        todos_resumos.append(resumo)

        plot_time_series(
            serie,
            janela,
            output_dir / f"comparativo_no_tempo_{janela.replace(' ', '_')}.png",
        )

        plot_final_values(
            resumo,
            janela,
            output_dir / f"valor_final_{janela.replace(' ', '_')}.png",
        )

        plot_returns(
            resumo,
            janela,
            output_dir / f"retorno_{janela.replace(' ', '_')}.png",
        )

    detalhe_final = pd.concat(todos_detalhes, ignore_index=True)
    resumo_final = pd.concat(todos_resumos, ignore_index=True)

    excel_path = output_dir / "comparativo_rankings_investidor10.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo_final.round(4).to_excel(writer, sheet_name="Resumo", index=False)
        detalhe_final.round(4).to_excel(writer, sheet_name="Detalhe por ativo", index=False)

        for janela, serie in todas_series.items():
            sheet = f"Serie {janela}".replace(" ", "_")[:31]
            serie.round(4).to_excel(writer, sheet_name=sheet)

    print("\nResumo:")
    print(
        resumo_final
        .sort_values(["Janela", "Retorno estimado (%)"], ascending=[True, False])
        .round(2)
        .to_string(index=False)
    )

    problemas = detalhe_final[detalhe_final["Status"] != "OK"].copy()
    if not problemas.empty:
        print("\nAtenção: alguns ativos tiveram dados ausentes ou parciais:")
        print(problemas[["Janela", "Carteira", "Ativo", "Yahoo", "Status"]].to_string(index=False))

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")


if __name__ == "__main__":
    main()
