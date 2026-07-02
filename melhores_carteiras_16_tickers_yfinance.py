"""
melhores_carteiras_16_tickers_yfinance.py

Objetivo:
A partir da lista de ações do PDF/prints do Investidor10, descobrir quais seriam
as carteiras de 16 tickers que melhor performaram em:

- 1 ano
- 3 anos
- 5 anos
- 10 anos

E comparar essas carteiras vencedoras com a sua carteira atual.

Premissas:
- Universo de ações: tickers extraídos da lista "Todas as Ações" enviada.
- Cada carteira tem 16 tickers.
- R$ 10.000 investidos em cada carteira.
- Pesos iguais: R$ 10.000 / 16 por ativo.
- Usa preço ajustado do Yahoo Finance quando disponível.
- Preço ajustado é uma proxy prática de retorno total, considerando ajustes
  por dividendos/JCP/desdobramentos.

Muito importante:
- Esse estudo é uma otimização olhando para trás.
- A "melhor carteira" de 10 anos, 5 anos, 3 anos ou 1 ano NÃO é recomendação de compra.
- Serve para identificar padrões, armadilhas, setores vencedores e ativos que
  teriam melhorado ou piorado sua carteira.
- Para 10 anos, muitos ativos atuais não existiam ou não tinham histórico completo.
  O script marca isso no diagnóstico.

Instalação:
    pip install yfinance pandas matplotlib openpyxl

Execução:
    python melhores_carteiras_16_tickers_yfinance.py

Saída:
    pasta saida_melhores_16_tickers/
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


VALOR_INICIAL = 10_000.00
N_TICKERS = 16

# =====================================================================
# Sua carteira atual
# =====================================================================
MINHA_CARTEIRA = [
    "CMIG3", "VALE3", "ITSA4", "CPLE3", "BBSE3", "PETR4", "GOAU4", "BBDC4",
    "ITUB4", "WEGE3", "TAEE11", "KLBN11", "BBAS3", "SAPR4", "ISAE3", "CSAN3"
]

# =====================================================================
# Universo de ações a partir do PDF "Todas as Ações" / prints enviados.
# Inclui as páginas do ranking por valor de mercado do Investidor10.
# =====================================================================
UNIVERSO_ACOES = [
    # Página 1
    "PETR4", "ITUB4", "VALE3", "ABEV3", "BPAC11", "WEGE3", "BBDC3",
    "AXIA3", "ITSA4", "BBAS3", "VIVT3", "SANB11", "SBSP3", "B3SA3",
    "RDOR3", "BBSE3", "PRIO3", "EMBJ3", "TIMS3", "CXSE3", "SUZB3",
    "CPFE3", "EQTL3", "ENEV3", "RENT3", "GGBR3",

    # Página 2
    "CPLE3", "EGIE3", "VBBR3", "CMIG4", "RADL3", "PSSA3", "MOTV3",
    "UGPA3", "RAIL3", "ENGI11", "CMIN3", "MBRF3", "KLBN11", "TOTS3",
    "CSMG3", "ISAE4", "PASS3", "CGAS3", "CSAN3", "REDE3", "LREN3",
    "HYPE3", "MULT3", "ALOS3", "NATU3", "MRSA3B", "TAEE11", "EQPA3",
    "USIM5", "CEEB3",

    # Página 3
    "GOAU4", "AURE3", "ASAI3", "SAPR11", "ALUP11", "SMFT3", "BNBR3",
    "CEGR3", "CYRE3", "BRAV3", "CURY3", "GMAT3", "ENMT4", "GGPS3",
    "CSNA3", "BRAP4", "EKTR3", "CASN3", "FLRY3", "BMEB4", "ORVR3",
    "JHSF3", "TTEN3", "SLCE3",
]

# Remove duplicados preservando ordem
UNIVERSO_ACOES = list(dict.fromkeys(UNIVERSO_ACOES))

# Para comparar com sua carteira, adiciono seus tickers caso não estejam no universo.
UNIVERSO_COM_MINHA = list(dict.fromkeys(UNIVERSO_ACOES + MINHA_CARTEIRA))


# =====================================================================
# Ajustes de ticker, se necessário.
# Caso o Yahoo não reconheça algum ticker, ajuste aqui.
# =====================================================================
TICKER_MAP = {
    # Exemplos de fallback, se necessário:
    # "MRSA3B": "MRSA3B",
    # "CGAS3": "CGAS3",
    # "ISAE3": "ISAE3",
}


def yf_symbol(ticker_b3: str) -> str:
    ticker = TICKER_MAP.get(ticker_b3, ticker_b3)
    return f"{ticker}.SA"


def get_price_series(symbol: str, start: pd.Timestamp) -> pd.Series:
    """
    Baixa série histórica do Yahoo Finance.
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


def evaluate_ticker(ticker: str, start: pd.Timestamp) -> dict:
    """
    Calcula retorno do ticker no período.
    Exige histórico próximo da data inicial para ser elegível.
    """
    symbol = yf_symbol(ticker)
    s = get_price_series(symbol, start=start)

    if s.empty:
        return {
            "Ativo": ticker,
            "Yahoo": symbol,
            "Elegível": False,
            "Status": "Sem dados",
            "Data inicial usada": None,
            "Data final usada": None,
            "Preço inicial ajustado (R$)": math.nan,
            "Preço atual ajustado (R$)": math.nan,
            "Retorno acumulado (%)": math.nan,
        }

    data_inicial_usada = pd.Timestamp(s.index[0]).normalize()
    data_final_usada = pd.Timestamp(s.index[-1]).normalize()
    dias_apos_inicio = (data_inicial_usada - start).days

    preco_inicial = float(s.iloc[0])
    preco_atual = float(s.iloc[-1])
    retorno = (preco_atual / preco_inicial - 1) * 100

    # Critério para evitar distorção: para cada janela,
    # o ativo precisa ter dados até 10 dias após a data inicial.
    elegivel = dias_apos_inicio <= 10

    if elegivel:
        status = "OK"
    else:
        status = f"Histórico parcial: começou {dias_apos_inicio} dias após início"

    return {
        "Ativo": ticker,
        "Yahoo": symbol,
        "Elegível": elegivel,
        "Status": status,
        "Data inicial usada": data_inicial_usada.date(),
        "Data final usada": data_final_usada.date(),
        "Preço inicial ajustado (R$)": preco_inicial,
        "Preço atual ajustado (R$)": preco_atual,
        "Retorno acumulado (%)": retorno,
    }


def build_portfolio_series(tickers: list[str], start: pd.Timestamp, nome: str) -> pd.Series:
    """
    Cria série histórica de carteira com pesos iguais em base 100.
    """
    normalized = []

    for ticker in tickers:
        s = get_price_series(yf_symbol(ticker), start=start)
        if s.empty:
            continue

        # Rejeita séries parciais no índice da carteira vencedora
        data_inicial_usada = pd.Timestamp(s.index[0]).normalize()
        if (data_inicial_usada - start).days > 10:
            continue

        n = normalize_100(s)
        if not n.empty:
            n.name = ticker
            normalized.append(n)

    if not normalized:
        return pd.Series(dtype=float, name=nome)

    df = pd.concat(normalized, axis=1).sort_index().ffill().dropna()
    carteira = df.mean(axis=1, skipna=True)
    carteira.name = nome
    return normalize_100(carteira)


def portfolio_value_from_returns(ranking: pd.DataFrame, tickers: list[str]) -> float:
    """
    Calcula valor atual de R$ 10.000 dividido igualmente entre tickers.
    """
    sub = ranking[ranking["Ativo"].isin(tickers)].copy()
    valor_por_ativo = VALOR_INICIAL / len(tickers)
    valores = valor_por_ativo * (1 + sub["Retorno acumulado (%)"] / 100)
    return float(valores.sum())


def summarize_portfolio(ranking: pd.DataFrame, tickers: list[str], nome: str, janela: str) -> dict:
    sub = ranking[ranking["Ativo"].isin(tickers)].copy()
    valor_atual = portfolio_value_from_returns(ranking, tickers)
    retorno = (valor_atual / VALOR_INICIAL - 1) * 100

    return {
        "Janela": janela,
        "Carteira": nome,
        "Ativos": len(tickers),
        "Ativos com retorno": sub["Retorno acumulado (%)"].notna().sum(),
        "Investimento inicial (R$)": VALOR_INICIAL,
        "Valor atual estimado (R$)": valor_atual,
        "Ganho/Perda estimado (R$)": valor_atual - VALOR_INICIAL,
        "Retorno estimado (%)": retorno,
    }


def compare_tickers(best_tickers: list[str], minha_tickers: list[str], janela: str) -> pd.DataFrame:
    best_set = set(best_tickers)
    minha_set = set(minha_tickers)

    rows = []

    for t in sorted(best_set & minha_set):
        rows.append({"Janela": janela, "Ativo": t, "Categoria": "Em ambas"})

    for t in sorted(best_set - minha_set):
        rows.append({"Janela": janela, "Ativo": t, "Categoria": "Só na melhor carteira"})

    for t in sorted(minha_set - best_set):
        rows.append({"Janela": janela, "Ativo": t, "Categoria": "Só na minha carteira"})

    return pd.DataFrame(rows)


def plot_bar_resumo(resumo: pd.DataFrame, output_path: Path) -> None:
    """
    Gráfico comparando melhor carteira vs minha carteira por janela.
    """
    pivot = resumo.pivot(index="Janela", columns="Carteira", values="Retorno estimado (%)")
    ordem = ["1 ano", "3 anos", "5 anos", "10 anos"]
    pivot = pivot.loc[[x for x in ordem if x in pivot.index]]

    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Melhor carteira de 16 tickers vs minha carteira")
    ax.set_ylabel("Retorno acumulado estimado (%)")
    ax.set_xlabel("Janela")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_time_comparison(series_dict: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for janela, df in series_dict.items():
        if df.empty:
            continue

        plt.figure(figsize=(13, 7))

        for col in df.columns:
            plt.plot(df.index, df[col], linewidth=2, label=col)

        plt.axhline(100, linestyle="--", linewidth=1)
        plt.title(f"Melhor carteira vs minha carteira - {janela}\nBase 100, pesos iguais")
        plt.xlabel("Data")
        plt.ylabel("Índice base 100")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"comparativo_no_tempo_{janela.replace(' ', '_')}.png", dpi=160)
        plt.close()


def plot_top16(ranking_top16: pd.DataFrame, janela: str, output_path: Path) -> None:
    data = ranking_top16.sort_values("Retorno acumulado (%)", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(data["Ativo"], data["Retorno acumulado (%)"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title(f"Top 16 tickers por performance - {janela}")
    plt.xlabel("Retorno acumulado estimado (%)")
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

    output_dir = Path("saida_melhores_16_tickers")
    output_dir.mkdir(exist_ok=True)

    todos_rankings = []
    todos_top16 = []
    todos_resumos = []
    todas_diferencas = []
    diagnosticos_minha = []
    series_para_plot = {}
    carteiras_vencedoras = {}

    for janela, start in janelas.items():
        print(f"\nProcessando janela: {janela}")

        # Avalia universo de ações para formar a melhor carteira
        ranking_rows = []
        for ticker in UNIVERSO_ACOES:
            ranking_rows.append(evaluate_ticker(ticker, start))

        ranking = pd.DataFrame(ranking_rows)
        ranking["Janela"] = janela
        todos_rankings.append(ranking)

        elegiveis = ranking[ranking["Elegível"] == True].copy()
        elegiveis = elegiveis.sort_values("Retorno acumulado (%)", ascending=False)

        top16 = elegiveis.head(N_TICKERS).copy()
        top16["Posição"] = range(1, len(top16) + 1)
        top16["Janela"] = janela
        todos_top16.append(top16)

        best_tickers = top16["Ativo"].tolist()
        carteiras_vencedoras[janela] = best_tickers

        # Avalia minha carteira na mesma janela
        minha_rows = []
        for ticker in MINHA_CARTEIRA:
            minha_rows.append(evaluate_ticker(ticker, start))
        minha_eval = pd.DataFrame(minha_rows)
        minha_eval["Janela"] = janela
        diagnosticos_minha.append(minha_eval)

        minha_elegivel = minha_eval[minha_eval["Elegível"] == True].copy()
        minha_tickers_elegiveis = minha_elegivel["Ativo"].tolist()

        # Resumo da carteira vencedora e da sua carteira
        resumo_best = summarize_portfolio(ranking, best_tickers, "Melhor 16 tickers", janela)
        resumo_minha = summarize_portfolio(minha_eval, minha_tickers_elegiveis, "Minha carteira", janela)

        todos_resumos.extend([resumo_best, resumo_minha])

        # Diferenças entre tickers
        diferencas = compare_tickers(best_tickers, MINHA_CARTEIRA, janela)
        todas_diferencas.append(diferencas)

        # Séries no tempo
        serie_best = build_portfolio_series(best_tickers, start, "Melhor 16 tickers")
        serie_minha = build_portfolio_series(minha_tickers_elegiveis, start, "Minha carteira")

        df_series = pd.concat([serie_best, serie_minha], axis=1).dropna()
        if not df_series.empty:
            df_series = df_series / df_series.iloc[0] * 100
        series_para_plot[janela] = df_series

        plot_top16(
            top16,
            janela,
            output_dir / f"top16_{janela.replace(' ', '_')}.png",
        )

    ranking_final = pd.concat(todos_rankings, ignore_index=True)
    top16_final = pd.concat(todos_top16, ignore_index=True)
    resumo_final = pd.DataFrame(todos_resumos)
    diferencas_final = pd.concat(todas_diferencas, ignore_index=True)
    minha_diag_final = pd.concat(diagnosticos_minha, ignore_index=True)

    # Diferença de tickers entre as melhores carteiras de cada janela
    diffs_janelas_rows = []
    janelas_lista = list(janelas.keys())

    for i in range(len(janelas_lista)):
        for j in range(i + 1, len(janelas_lista)):
            j1 = janelas_lista[i]
            j2 = janelas_lista[j]
            s1 = set(carteiras_vencedoras[j1])
            s2 = set(carteiras_vencedoras[j2])

            for t in sorted(s1 & s2):
                diffs_janelas_rows.append({
                    "Comparação": f"{j1} vs {j2}",
                    "Ativo": t,
                    "Categoria": "Em ambas as melhores",
                })

            for t in sorted(s1 - s2):
                diffs_janelas_rows.append({
                    "Comparação": f"{j1} vs {j2}",
                    "Ativo": t,
                    "Categoria": f"Só na melhor de {j1}",
                })

            for t in sorted(s2 - s1):
                diffs_janelas_rows.append({
                    "Comparação": f"{j1} vs {j2}",
                    "Ativo": t,
                    "Categoria": f"Só na melhor de {j2}",
                })

    diffs_janelas = pd.DataFrame(diffs_janelas_rows)

    # Gráficos gerais
    plot_bar_resumo(resumo_final, output_dir / "resumo_melhor_vs_minha.png")
    plot_time_comparison(series_para_plot, output_dir)

    excel_path = output_dir / "melhores_carteiras_16_tickers.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        resumo_final.round(4).to_excel(writer, sheet_name="Resumo", index=False)
        top16_final.round(4).to_excel(writer, sheet_name="Melhores 16 por janela", index=False)
        diferencas_final.to_excel(writer, sheet_name="Diferenças vs minha", index=False)
        diffs_janelas.to_excel(writer, sheet_name="Diferenças entre janelas", index=False)
        minha_diag_final.round(4).to_excel(writer, sheet_name="Minha carteira detalhe", index=False)
        ranking_final.round(4).to_excel(writer, sheet_name="Ranking universo", index=False)

        for janela, df in series_para_plot.items():
            sheet = f"Serie {janela}".replace(" ", "_")[:31]
            df.round(4).to_excel(writer, sheet_name=sheet)

    # Arquivo txt mais fácil de ler no terminal
    txt_path = output_dir / "resumo_tickers.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("MELHORES CARTEIRAS DE 16 TICKERS POR JANELA\n")
        f.write("=" * 60 + "\n\n")

        for janela in janelas.keys():
            f.write(f"{janela.upper()}\n")
            f.write("-" * 60 + "\n")
            f.write("Melhor carteira:\n")
            f.write(", ".join(carteiras_vencedoras[janela]) + "\n\n")

            diff = diferencas_final[diferencas_final["Janela"] == janela]
            so_melhor = diff[diff["Categoria"] == "Só na melhor carteira"]["Ativo"].tolist()
            so_minha = diff[diff["Categoria"] == "Só na minha carteira"]["Ativo"].tolist()
            ambas = diff[diff["Categoria"] == "Em ambas"]["Ativo"].tolist()

            f.write("Em ambas:\n")
            f.write(", ".join(ambas) + "\n\n")

            f.write("Só na melhor carteira:\n")
            f.write(", ".join(so_melhor) + "\n\n")

            f.write("Só na minha carteira:\n")
            f.write(", ".join(so_minha) + "\n\n\n")

    print("\nResumo:")
    print(
        resumo_final
        .sort_values(["Janela", "Carteira"])
        .round(2)
        .to_string(index=False)
    )

    print("\nMelhores carteiras:")
    for janela, tickers in carteiras_vencedoras.items():
        print(f"{janela}: {', '.join(tickers)}")

    problemas = pd.concat([ranking_final, minha_diag_final], ignore_index=True)
    problemas = problemas[problemas["Status"] != "OK"].copy()

    if not problemas.empty:
        print("\nAtenção: alguns ativos tiveram dados ausentes ou histórico parcial.")
        print("Veja as abas 'Ranking universo' e 'Minha carteira detalhe' no Excel.")

    print(f"\nArquivos gerados em: {output_dir.resolve()}")
    print(f"Excel: {excel_path.resolve()}")
    print(f"Resumo de tickers: {txt_path.resolve()}")


if __name__ == "__main__":
    main()
