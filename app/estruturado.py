"""Saída estruturada: o modelo preenche um JSON por esquema e o CÓDIGO monta o texto da resposta.

Por quê: o formato do prompt v2.0 (seção "Relatos de experiência", linha "Fundamentado nas referências", relato só na seção
própria) era pedido em linguagem natural e os modelos pequenos o ignoravam; o 4B de produção também citava relato fora da
seção. Aqui o formato vem do código:
  * `esquema` diz ao Ollama (parâmetro `format`) quais campos existem e, em cada `refs`, quais números podem ser citados:
    as seções de referência só citam referências e cada relato só cita relatos, então "relato fora da seção" e "[n]
    inexistente" ficam impossíveis por construção;
  * `renderizar` monta o MESMO markdown do prompt v2.0 (títulos, linha de fundamentação, lista de fontes) a partir dos campos.
O harness de saída continua rodando sobre o texto montado: é a segunda camada, e nenhuma regra dele foi relaxada.

Limites de tamanho do esquema são só teto de segurança: com `maxLength` apertado o Ollama corta a frase no meio. O limite de
350 palavras fica no prompt e vira alerta no harness.
"""
from __future__ import annotations

import json
import re

from .config import REFERENCIAS
from .indice import grupo_do_doc, rotulo_pagina
from .prompts import ETIQUETA_GRUPO, PREFIXO_FUNDAMENTACAO, ROTULOS_FUNDAMENTACAO, _cabecalho

SECOES = (("orientacao", "Orientação"), ("atividade", "Atividade em sala"), ("acompanhamento", "Acompanhamento"))
AVISO_SO_RELATOS = "Esta orientação vem de relatos de experiência e não foi encontrada nas referências da base."
NAO_ENCONTRADO = ROTULOS_FUNDAMENTACAO[2]  # "Não encontrado nas referências"

MIN_TEXTO = 20     # recusa campo vazio ou de uma palavra
MAX_TEXTO = 1200   # teto de segurança contra texto sem fim, não limite de estilo

_CITA_LIVRE = re.compile(r"\s*\[\d{1,2}(?:\s*[,;]\s*\d{1,2})*\]")
_CERCA = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.I)


class EstruturadoInvalido(ValueError):
    """O JSON existe, mas não dá para montar uma resposta íntegra com ele."""


def _indices(modo: str, trechos: list[dict]) -> tuple[list[int], list[int]]:
    """(números das referências, números dos relatos) na ordem da janela; [n] é a posição + 1."""
    todos = list(range(1, len(trechos) + 1))
    if modo in ("referencia", "unico"):  # fluxo sem relatos separados (ou antigo, CASCATA=0): toda fonte é referência
        return todos, []
    if modo == "experiencia":
        return [], todos
    refs = [i for i in todos if grupo_do_doc(trechos[i - 1]["doc"]) in REFERENCIAS]
    return refs, [i for i in todos if i not in refs]


def modo_efetivo(modo: str, trechos: list[dict]) -> str:
    """Modo misto sem nenhum relato na janela (orçamento esgotado) vira só referências."""
    if modo == "referencia+experiencia" and not _indices(modo, trechos)[1]:
        return "referencia"
    return modo


def _refs(validos: list[int], minimo: int, maximo: int) -> dict:
    return {"type": "array", "items": {"type": "integer", "enum": validos}, "minItems": minimo, "maxItems": maximo}


def _secao(refs: list[int]) -> dict:
    return {"type": "object", "properties": {
        "texto": {"type": "string", "minLength": MIN_TEXTO, "maxLength": MAX_TEXTO},
        "refs": _refs(refs, 1, 3),
    }, "required": ["texto", "refs"]}


def _relato(refs: list[int], relatos: list[int], com_fundamentacao: bool) -> dict:
    props = {
        "canal": {"type": "string", "minLength": 2, "maxLength": 80},
        "texto": {"type": "string", "minLength": MIN_TEXTO, "maxLength": MAX_TEXTO},
        "refs": _refs(relatos, 1, 1),
    }
    if com_fundamentacao:
        props["fundamentacao"] = {"type": "string", "enum": list(ROTULOS_FUNDAMENTACAO)}
        props["refs_fundamentacao"] = _refs(refs, 0, 3)
    return {"type": "object", "properties": props, "required": list(props)}


def esquema(modo: str, trechos: list[dict]) -> dict:
    """JSON schema do `format` do Ollama para o modo da tabela de decisão e a janela de trechos."""
    modo = modo_efetivo(modo, trechos)
    refs, relatos = _indices(modo, trechos)
    props: dict = {}
    if modo != "experiencia":
        for chave, _ in SECOES:
            props[chave] = _secao(refs)
    if relatos and modo in ("referencia+experiencia", "experiencia"):
        props["relatos"] = {"type": "array", "minItems": 1, "maxItems": len(relatos),
                            "items": _relato(refs, relatos, com_fundamentacao=modo == "referencia+experiencia")}
    return {"type": "object", "properties": props, "required": list(props)}


def parse(bruto: str) -> dict | None:
    """JSON da resposta do modelo, ou None. Tolera cerca de código; JSON truncado é inválido (o pipeline tenta de novo)."""
    try:
        dados = json.loads(_CERCA.sub("", bruto.strip()))
    except (json.JSONDecodeError, ValueError):
        return None
    return dados if isinstance(dados, dict) else None


def _limpar(valor: object) -> str:
    """Texto do campo sem [n] soltos: quem cita é o campo `refs`, que o esquema restringe."""
    return re.sub(r"[ \t]{2,}", " ", _CITA_LIVRE.sub("", str(valor or ""))).strip()


def _nums(valor: object, validos: list[int]) -> list[int]:
    saida: list[int] = []
    for x in valor if isinstance(valor, list) else []:
        try:
            n = int(x)
        except (TypeError, ValueError):
            continue
        if n in validos and n not in saida:
            saida.append(n)
    return saida


def _marca(nums: list[int]) -> str:
    return "".join(f"[{n}]" for n in nums)


_SEM_SUJEITO = {"a", "o", "as", "os", "um", "uma", "ele", "ela", "eles", "elas"}


def _com_canal_como_sujeito(canal: str, texto: str) -> str:
    """O prompt pede o canal como sujeito da frase; um modelo pequeno às vezes só devolve a ação ("a turma lê em duplas").
    Só acrescenta o canal quando o texto não traz sujeito próprio (nome, "Segundo…", "Outro relato…")."""
    if not canal or canal.lower() in texto.lower():
        return texto
    primeira = texto.split(None, 1)[0].strip(",.;:").lower()
    if texto[0].isupper() and primeira not in _SEM_SUJEITO:
        return texto
    return f"{canal} relata que {texto[0].lower()}{texto[1:]}"


def _linha_fundamentacao(rotulo: str, refs_fund: list[int], trechos: list[dict]) -> str:
    if rotulo == NAO_ENCONTRADO or not refs_fund:
        return f"*{PREFIXO_FUNDAMENTACAO} {NAO_ENCONTRADO}*"
    por_tipo: dict[str, list[int]] = {}
    for n in refs_fund:  # o tipo vem do grupo da fonte citada, nunca do que o modelo escreveu
        por_tipo.setdefault(ETIQUETA_GRUPO[grupo_do_doc(trechos[n - 1]["doc"])], []).append(n)
    tipos = ", ".join(f"{tipo} {_marca(nums)}" for tipo, nums in por_tipo.items())
    return f"*{PREFIXO_FUNDAMENTACAO} {rotulo} — {tipos}*"


def renderizar(dados: dict, trechos: list[dict], modo: str, rotulos: dict[str, str]) -> str:
    """Monta o texto no formato do prompt v2.0. Levanta EstruturadoInvalido se faltar conteúdo ou fonte válida."""
    modo = modo_efetivo(modo, trechos)
    refs, relatos = _indices(modo, trechos)
    partes: list[str] = []
    citadas: set[int] = set()

    if modo == "experiencia":
        partes.append(AVISO_SO_RELATOS)
    else:
        for chave, titulo in SECOES:
            sec = dados.get(chave)
            if not isinstance(sec, dict):
                raise EstruturadoInvalido(f"campo ausente: {chave}")
            texto, nums = _limpar(sec.get("texto")), _nums(sec.get("refs"), refs)
            if len(texto) < MIN_TEXTO:
                raise EstruturadoInvalido(f"{chave}: texto vazio ou curto demais")
            if not nums:
                raise EstruturadoInvalido(f"{chave}: nenhuma fonte válida em refs")
            citadas.update(nums)
            partes.append(f"**{titulo}**\n{texto} {_marca(nums)}")

    if relatos:
        itens = dados.get("relatos")
        if not isinstance(itens, list):
            raise EstruturadoInvalido("campo ausente: relatos")
        paragrafos: list[str] = []
        usados: set[int] = set()
        for it in itens:
            if not isinstance(it, dict):
                continue
            nums = _nums(it.get("refs"), relatos)
            texto = _limpar(it.get("texto"))
            if not nums or len(texto) < MIN_TEXTO or usados & set(nums):
                continue
            usados.update(nums)
            texto = _com_canal_como_sujeito(_limpar(it.get("canal")), texto)
            rotulo = it.get("fundamentacao") if it.get("fundamentacao") in ROTULOS_FUNDAMENTACAO else NAO_ENCONTRADO
            fund = _nums(it.get("refs_fundamentacao"), refs) if modo == "referencia+experiencia" else []
            citadas.update(nums + fund)
            paragrafos.append(f"{texto} {_marca(nums)}\n{_linha_fundamentacao(rotulo, fund, trechos)}")
        if not paragrafos:
            raise EstruturadoInvalido("relatos: nenhum relato utilizável")
        partes.append("**Relatos de experiência**\n" + "\n\n".join(paragrafos))

    linhas = []
    for n in sorted(citadas):
        c = trechos[n - 1]
        if grupo_do_doc(c["doc"]) == "experiencia" and modo != "unico":
            linhas.append(_cabecalho(n, c, rotulos, True))
        else:
            linhas.append(f"[{n}] {rotulos.get(c['doc'], c['doc'])}, {rotulo_pagina(c)}")
    partes.append("**Fontes**\n" + "\n".join(linhas))
    return "\n\n".join(partes)
