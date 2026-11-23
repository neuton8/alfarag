"""Harness v2.1: guarda de escopo, código BNCC malformado e alertas de tamanho e autor-ano. Só acrescentam regras."""
import csv

import pytest

from app import harness
from app.config import RAIZ


def _gold():
    return list(csv.DictReader((RAIZ / "benchmark" / "perguntas_gold.csv").open(encoding="utf-8")))


def _limites():
    return {x["id"]: x["question"] for x in csv.DictReader((RAIZ / "benchmark" / "casos_limite.csv").open(encoding="utf-8"))}


def _trecho(texto, doc="BNCC", pagina=3):
    return {"id": f"{doc}-{pagina}", "doc": doc, "texto": texto, "tokens": 20, "secao": "", "pagina_ini": pagina, "pagina_fim": pagina}


# ---------- guarda de escopo ----------
@pytest.mark.parametrize("pergunta", [
    "Como ensinar frações equivalentes para uma turma do 5º ano?",
    "Como ensinar fracoes para o 5o ano?",
    "Como ensinar equações no 8º ano?",
    "Qual a melhor forma de ensinar geometria no ensino médio?",
    "Como ensinar a tabuada para o quarto ano?",
    "Como trabalhar matemática com a turma do 6º ano?",
])
def test_fora_do_escopo(pergunta):
    assert harness.fora_do_escopo(pergunta)
    assert harness.checar_entrada(pergunta).fora_escopo


@pytest.mark.parametrize("pergunta", [
    "Tenho um aluno do 5º ano que ainda não lê palavras simples. Como alfabetizá-lo?",  # alfabetização tardia
    "Meu aluno de 7 anos troca letras ao escrever. O que fazer?",                       # idade não é série
    "Como ensinar a ler frações em um enunciado de problema para o 3º ano?",            # 3º ano e leitura
    "Como ensinar matemática com letras móveis para o 2º ano?",                         # fala de 2º ano
    "Como planejar a rotina de leitura do 1º ao 5º ano?",                               # cita o 1º
    "Como ensinar sílabas?",
])
def test_dentro_do_escopo(pergunta):
    assert harness.fora_do_escopo(pergunta) == ""
    assert harness.checar_entrada(pergunta).fora_escopo == ""


def test_guarda_nao_marca_nenhuma_pergunta_gold():
    marcadas = [g["id"] for g in _gold() if harness.fora_do_escopo(g["question"])]
    assert len(_gold()) == 24 and marcadas == []


def test_guarda_pega_o_L02_e_nenhum_outro_caso_limite():
    marcados = [i for i, q in _limites().items() if harness.fora_do_escopo(q)]
    assert marcados == ["L02"]  # L03 (receita de bolo) continua sendo recusado pelo limiar de cobertura, como antes


def test_guarda_nao_atrapalha_os_bloqueios_existentes():
    e = harness.checar_entrada(_limites()["L04"])  # CPF
    assert not e.ok and e.motivo == "dado_pessoal"
    e = harness.checar_entrada(_limites()["L01"])  # diagnóstico
    assert e.ok and e.pede_diagnostico and e.fora_escopo == ""


# ---------- código BNCC malformado ----------
def test_codigo_bncc_malformado_e_removido():
    trechos = [_trecho("A habilidade EF02LP01 trata da leitura de palavras.")]
    texto, v = harness.verificar_saida("Trabalhe a habilidade EF02LP4 com a turma [1].", trechos, False, "stop")
    assert v.codigos_removidos == ["EF02LP4"] and "EF02LP4" not in texto and "[código BNCC removido" in texto


def test_codigo_bncc_correto_continua_valido():
    trechos = [_trecho("As habilidades EF02LP01 e EF02LP01A tratam de leitura.")]
    texto, v = harness.verificar_saida("Trabalhe EF02LP01 e EF02LP01A [1].", trechos, False, "stop")
    assert v.codigos_removidos == [] and v.codigos_ok == ["EF02LP01", "EF02LP01A"]


# ---------- palavras ----------
def test_alerta_de_excesso_de_palavras():
    resposta = "**Orientação**\n" + ("palavra " * 400) + "[1]"
    texto, v = harness.verificar_saida(resposta, [_trecho("Texto.")], False, "stop")
    assert v.excesso_palavras and v.palavras > 350 and any("350" in a for a in v.alertas) and texto == resposta  # só alerta


def test_fontes_nao_contam_no_limite_de_palavras():
    resposta = "**Orientação**\n" + ("palavra " * 100) + "[1]\n\n**Fontes**\n" + ("[1] Manual ABC (2021), p. 519 " * 100)
    _, v = harness.verificar_saida(resposta, [_trecho("Texto.")], False, "stop")
    assert not v.excesso_palavras and v.palavras < 150


# ---------- autor-ano ----------
def test_autor_ano_fora_das_fontes_gera_alerta_sem_mexer_no_texto():
    resposta = "Segundo Silva (2020), a rima ajuda a ler [1]. Para Costa et al. (2019) também [1]."
    texto, v = harness.verificar_saida(resposta, [_trecho("A rima ajuda a alfabetização.")], False, "stop")
    assert v.autores_fora_das_fontes == ["Silva (2020)", "Costa (2019)"] and texto == resposta
    assert any("Silva (2020)" in a for a in v.alertas)


def test_autor_ano_presente_no_texto_ou_no_rotulo_da_fonte_nao_alerta():
    trechos = [_trecho("Hall e colegas analisaram 40 estudos.", doc="HALL"), _trecho("Texto sem autores.", doc="RAMLI")]
    rotulos = {"HALL": "Hall e Silva e Silva (2019)", "RAMLI": "Ramli et al. (2021)"}
    resposta = "Hall et al. (2019) e (Ramli, 2021) apontam efeito moderado [1][2]. Silva, 2019 concorda [1]."
    _, v = harness.verificar_saida(resposta, trechos, False, "stop", None, rotulos)
    assert v.autores_fora_das_fontes == []


def test_autor_ano_ignora_a_lista_de_fontes():
    resposta = "Rimas ajudam [1].\n\n**Fontes**\n[1] Autor Desconhecido (2020), p. 3"
    _, v = harness.verificar_saida(resposta, [_trecho("Rimas.")], False, "stop")
    assert v.autores_fora_das_fontes == []


def test_nomes_comuns_seguidos_de_ano_nao_viram_citacao_falsa():
    _, v = harness.verificar_saida("A BNCC (2018) e o Manual ABC (2021) orientam [1]. Em 2020 a escola mudou [1].",
                                   [_trecho("BNCC e Manual ABC.")], False, "stop")
    assert v.autores_fora_das_fontes == []
