"""Prompt Engineering — a instrução que o modelo recebe, versionada.

O texto do PROMPT_SISTEMA é o da Etapa 4 do projeto do grupo, acrescido das
regras de formato que o harness consegue verificar depois (citação [n]).
"""
from __future__ import annotations

import re

from .indice import grupo_do_doc, rotulo_pagina

# v2.0: fontes separadas por grupo (documentos oficiais, produção científica, relatos de experiência) e
# instrução por modo da tabela de decisão (seção "Relatos de experiência", identificada pelo canal, com a linha
# "Fundamentado nas referências: Sim / Em parte — <tipo> [n] / Não encontrado nas referências").
# v2.1: acrescenta a variante em JSON por esquema (PROMPT_SISTEMA_JSON, montar_mensagens_json): o modelo preenche campos e o
# código monta o texto (app/estruturado.py). O texto livre da v2.0 continua igual e é usado com ESTRUTURADO=0 e como reserva.
VERSAO_PROMPT = "v2.1"

PROMPT_SISTEMA = """Você é um assistente pedagógico especializado em alfabetização em língua portuguesa brasileira, destinado a professores do 2º e 3º anos do Ensino Fundamental. Responda exclusivamente com base nos documentos recuperados da base de conhecimento, que aparecem numerados entre colchetes no bloco FONTES. Priorize a BNCC para habilidades curriculares e os artigos científicos e o Manual ABC para evidências sobre intervenções em leitura e apoio a estudantes com dislexia.

Sempre que possível, organize a resposta em quatro partes curtas, com estes títulos:
**Orientação** — uma orientação objetiva.
**Atividade em sala** — uma atividade prática aplicável em sala de aula.
**Acompanhamento** — uma forma de acompanhar a aprendizagem.
**Fontes** — os documentos e páginas usados.

Regras obrigatórias:
1. Toda afirmação baseada em um trecho deve terminar com o número da fonte entre colchetes, por exemplo [1] ou [2][3]. Use somente números que existem no bloco FONTES.
2. Não invente códigos da BNCC (como EF02LP04), referências bibliográficas ou resultados científicos. Só cite um código BNCC se ele aparecer escrito no trecho.
3. Não transfira automaticamente exemplos de alfabetização em inglês para o português brasileiro; se usar um artigo em inglês, diga que a evidência vem de estudos em inglês.
4. Não realize diagnósticos de dislexia, TDAH, TEA ou outros transtornos. Oriente a observação pedagógica, o registro e o encaminhamento profissional quando necessário.
5. Quando os documentos não contiverem evidências suficientes, diga explicitamente: "Os documentos da base não trazem evidência suficiente sobre isso."
6. A decisão pedagógica final cabe ao professor e à equipe escolar.
Responda em português do Brasil, em no máximo 350 palavras."""

AVISO_DIAGNOSTICO = (
    "ATENÇÃO DO SISTEMA: a pergunta pede ou sugere um diagnóstico clínico. "
    "Comece a resposta dizendo, em uma frase, que o assistente não faz diagnóstico e que só uma avaliação "
    "profissional especializada pode concluir isso; depois ofereça apenas orientação pedagógica e de encaminhamento."
)


# Linha de fundamentação de cada relato: "Fundamentado nas referências: Sim — documento oficial [n]" etc.
PREFIXO_FUNDAMENTACAO = "Fundamentado nas referências:"
ROTULOS_FUNDAMENTACAO = ("Sim", "Em parte", "Não encontrado nas referências")

# Blocos de FONTES: referências (oficiais e científicas juntas, por relevância) e, depois, relatos.
# A numeração [n] segue a ordem dos trechos, que o pipeline já entrega nessa sequência.
BLOCO_DO_GRUPO = {"oficial": "referencia", "cientifico": "referencia", "experiencia": "experiencia"}
TITULOS_BLOCO = {
    "referencia": "REFERÊNCIAS (documentos oficiais e produção científica, em ordem de relevância)",
    "experiencia": "RELATOS DE EXPERIÊNCIA (resumos de vídeos em que educadores contam a própria prática; não são evidência)",
}
ETIQUETA_GRUPO = {"oficial": "documento oficial", "cientifico": "produção científica"}

_COMO_ESCREVER_RELATO = (
    "Para cada relato usado, escreva um parágrafo em que o canal indicado no cabeçalho do relato é o sujeito da "
    "frase, variando o verbo, por exemplo: \"Ludimila Lima relata que…\", \"Segundo Prof. Nágida, …\", "
    "\"O canal Nova Escola mostra que…\", \"Outro relato, de <canal>, descreve…\". "
    "Não use pronomes nem palavras com gênero (ela, ele, a professora, o professor) para se referir a quem fala: "
    "repita o nome do canal se precisar. Descreva o que foi relatado (\"conduz a turma…\", \"usa figuras…\"), "
    "sem dar ordens nem recomendações (evite \"deve\", \"é preciso\"), e termine o parágrafo com o número do relato "
    "entre colchetes."
)

INSTRUCAO_MISTO = (
    "ATENÇÃO DO SISTEMA: as FONTES trazem REFERÊNCIAS (documentos oficiais e produção científica) e "
    "RELATOS DE EXPERIÊNCIA (resumos de vídeos em que educadores contam a própria prática).\n"
    "- Baseie Orientação, Atividade em sala e Acompanhamento somente nas REFERÊNCIAS.\n"
    "- Depois de Acompanhamento e antes de Fontes, inclua a parte **Relatos de experiência**. "
    + _COMO_ESCREVER_RELATO + "\n"
    "- Logo abaixo de cada parágrafo, escreva uma linha em itálico (entre asteriscos) em um destes formatos, "
    "exatamente assim:\n"
    f"  *{PREFIXO_FUNDAMENTACAO} Sim — documento oficial [n]*\n"
    f"  *{PREFIXO_FUNDAMENTACAO} Sim — produção científica [n]*\n"
    f"  *{PREFIXO_FUNDAMENTACAO} Em parte — documento oficial [n]*\n"
    f"  *{PREFIXO_FUNDAMENTACAO} Em parte — produção científica [n]*\n"
    f"  *{PREFIXO_FUNDAMENTACAO} Não encontrado nas referências*\n"
    "  Use Sim (a referência sustenta o relato) ou Em parte (sustenta só parte dele) somente citando o número [n] "
    "de uma REFERÊNCIA; o tipo (documento oficial ou produção científica) está entre parênteses no cabeçalho dela. "
    "Se referências dos dois tipos sustentarem o relato, escreva os dois tipos, cada um com o seu número. "
    "Se nenhuma referência sustentar o relato, use Não encontrado nas referências."
)

INSTRUCAO_EXPERIENCIA = (
    "ATENÇÃO DO SISTEMA: nenhuma referência da base (documento oficial ou produção científica) cobre esta pergunta; "
    "as FONTES trazem só RELATOS DE EXPERIÊNCIA (resumos de vídeos em que educadores contam a própria prática).\n"
    "- Comece a resposta com a frase: \"Esta orientação vem de relatos de experiência e não foi encontrada nas "
    "referências da base.\"\n"
    "- Em vez de Orientação, Atividade em sala e Acompanhamento, escreva a parte **Relatos de experiência**. "
    + _COMO_ESCREVER_RELATO + "\n"
    f"- Logo abaixo de cada parágrafo, escreva a linha em itálico: *{PREFIXO_FUNDAMENTACAO} Não encontrado nas referências*\n"
    "- Termine com a parte Fontes."
)

_VIDEO_RE = re.compile(r"^vídeo (\S+), (.+?), (“.*”)$")


def _cabecalho(i: int, c: dict, rotulos: dict[str, str], por_grupo: bool = False) -> str:
    grupo = grupo_do_doc(c["doc"])
    if por_grupo and grupo == "experiencia":  # resumo de vídeo: sem página; a seção identifica vídeo, canal e título
        secao = c.get("secao") or ""
        m = _VIDEO_RE.match(secao)
        if m:  # "vídeo v004, Ludimila Lima, “Título”" -> canal explícito para o modelo citar
            return f"[{i}] Relato de experiência — canal: {m.group(2)} — vídeo {m.group(1)}, {m.group(3)}"
        return f"[{i}] Relato de experiência — {secao or rotulos.get(c['doc'], c['doc'])}"
    cab = f"[{i}] {rotulos.get(c['doc'], c['doc'])}, {rotulo_pagina(c)}"
    if c.get("secao"):
        cab += f" — {c['secao']}"
    if por_grupo:
        cab += f" ({ETIQUETA_GRUPO[grupo]})"
    return cab


def bloco_fontes(trechos: list[dict], rotulos: dict[str, str], por_grupo: bool = False) -> str:
    """Lista numerada de trechos. Com por_grupo=True, insere o título do bloco (REFERÊNCIAS / RELATOS) quando ele
    muda; os trechos já chegam com as referências antes dos relatos, então a numeração continua sequencial."""
    linhas = []
    bloco_atual = None
    for i, c in enumerate(trechos, 1):
        if por_grupo:
            b = BLOCO_DO_GRUPO[grupo_do_doc(c["doc"])]
            if b != bloco_atual:
                linhas.append(f"{TITULOS_BLOCO[b]}:")
                bloco_atual = b
        linhas.append(f"{_cabecalho(i, c, rotulos, por_grupo)}\n{c['texto']}")
    return "\n\n".join(linhas)


def montar_mensagens(pergunta: str, trechos: list[dict], rotulos: dict[str, str], pede_diagnostico: bool,
                     modo: str | None = None) -> list[dict]:
    """`modo` vem da tabela de decisão do pipeline. None ou "unico" (fluxo antigo, CASCATA=0): formato antigo."""
    por_grupo = modo in ("referencia", "referencia+experiencia", "experiencia")
    usuario = f"FONTES:\n{bloco_fontes(trechos, rotulos, por_grupo)}\n\nPERGUNTA DO PROFESSOR:\n{pergunta}"
    if modo == "referencia+experiencia":
        usuario += f"\n\n{INSTRUCAO_MISTO}"
    elif modo == "experiencia":
        usuario += f"\n\n{INSTRUCAO_EXPERIENCIA}"
    if pede_diagnostico:
        usuario += f"\n\n{AVISO_DIAGNOSTICO}"
    return [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": usuario},
    ]


# ---------- v2.1: variante em JSON por esquema ----------
# O formato (títulos, linha de fundamentação, lista de fontes) deixa de ser pedido em linguagem natural: o modelo só preenche
# campos, o esquema limita quais números cada campo pode citar e app/estruturado.py monta o texto final.
PROMPT_SISTEMA_JSON = """Você é um assistente pedagógico especializado em alfabetização em língua portuguesa brasileira, destinado a professores do 2º e 3º anos do Ensino Fundamental. Responda exclusivamente com base nos documentos recuperados da base de conhecimento, que aparecem numerados entre colchetes no bloco FONTES. Priorize a BNCC para habilidades curriculares e os artigos científicos e o Manual ABC para evidências sobre intervenções em leitura e apoio a estudantes com dislexia.

Você responde preenchendo um formulário em JSON; o sistema monta o texto final a partir dele.

Regras obrigatórias:
1. Em cada campo, escreva em "texto" o conteúdo em si, em frases completas em português do Brasil, sem repetir a pergunta do professor. Em "refs", liste apenas os números das fontes que sustentam aquele texto. Não escreva números entre colchetes dentro do texto.
2. Não invente códigos da BNCC (como EF02LP04), referências bibliográficas ou resultados científicos. Só cite um código BNCC se ele aparecer escrito no trecho.
3. Não transfira automaticamente exemplos de alfabetização em inglês para o português brasileiro; se usar um artigo em inglês, diga que a evidência vem de estudos em inglês.
4. Não realize diagnósticos de dislexia, TDAH, TEA ou outros transtornos. Oriente a observação pedagógica, o registro e o encaminhamento profissional quando necessário.
5. Quando os documentos não contiverem evidências suficientes, diga explicitamente: "Os documentos da base não trazem evidência suficiente sobre isso."
6. A decisão pedagógica final cabe ao professor e à equipe escolar.
Somando todos os campos, use no máximo 350 palavras."""

_CAMPOS_BASE = (
    'Preencha o JSON com: "orientacao" (uma orientação objetiva), "atividade" (uma atividade prática aplicável em sala '
    'de aula) e "acompanhamento" (uma forma de acompanhar a aprendizagem). Cada um tem "texto" e "refs" (números das fontes usadas).'
)

INSTRUCAO_JSON_REFERENCIA = _CAMPOS_BASE

INSTRUCAO_JSON_MISTO = (
    _CAMPOS_BASE + " Baseie esses três campos somente nas REFERÊNCIAS.\n"
    'Preencha também "relatos": um item para cada RELATO DE EXPERIÊNCIA que você usar, com\n'
    '- "canal": o nome do canal indicado no cabeçalho do relato;\n'
    '- "texto": um parágrafo em que o canal é o sujeito da frase, variando o verbo ("Ludimila Lima relata que…", '
    '"Segundo Prof. Nágida, …"), descrevendo o que foi relatado, sem dar ordens (evite "deve", "é preciso") e sem '
    "pronomes nem palavras com gênero para quem fala (ela, ele, a professora, o professor);\n"
    '- "refs": o número do relato;\n'
    '- "fundamentacao": "Sim" se uma REFERÊNCIA sustenta o relato, "Em parte" se sustenta só parte dele, '
    '"Não encontrado nas referências" se nenhuma sustenta;\n'
    '- "refs_fundamentacao": os números das REFERÊNCIAS que sustentam o relato (vazio se não encontrado).'
)

INSTRUCAO_JSON_EXPERIENCIA = (
    "ATENÇÃO DO SISTEMA: nenhuma referência da base cobre esta pergunta; as FONTES trazem só RELATOS DE EXPERIÊNCIA "
    '(resumos de vídeos em que educadores contam a própria prática). Preencha só "relatos": um item para cada relato usado, com\n'
    '- "canal": o nome do canal indicado no cabeçalho do relato;\n'
    '- "texto": um parágrafo em que o canal é o sujeito da frase, variando o verbo, descrevendo o que foi relatado, sem dar '
    'ordens (evite "deve", "é preciso") e sem pronomes nem palavras com gênero para quem fala (ela, ele, a professora, o professor);\n'
    '- "refs": o número do relato.'
)

AVISO_DIAGNOSTICO_JSON = AVISO_DIAGNOSTICO + " Escreva essa frase no começo do primeiro campo de texto."


def montar_mensagens_json(pergunta: str, trechos: list[dict], rotulos: dict[str, str], pede_diagnostico: bool,
                          modo: str | None = None) -> list[dict]:
    """Mesmas FONTES do texto livre; muda só o que se pede ao modelo (campos do JSON, não markdown)."""
    por_grupo = modo in ("referencia", "referencia+experiencia", "experiencia")
    usuario = f"FONTES:\n{bloco_fontes(trechos, rotulos, por_grupo)}\n\nPERGUNTA DO PROFESSOR:\n{pergunta}"
    if modo == "referencia+experiencia":
        usuario += f"\n\n{INSTRUCAO_JSON_MISTO}"
    elif modo == "experiencia":
        usuario += f"\n\n{INSTRUCAO_JSON_EXPERIENCIA}"
    else:
        usuario += f"\n\n{INSTRUCAO_JSON_REFERENCIA}"
    if pede_diagnostico:
        usuario += f"\n\n{AVISO_DIAGNOSTICO_JSON}"
    return [
        {"role": "system", "content": PROMPT_SISTEMA_JSON},
        {"role": "user", "content": usuario},
    ]
