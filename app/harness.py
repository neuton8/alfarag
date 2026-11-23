"""Harness — regras determinísticas em volta do modelo.

Antes do modelo (o que o sistema impede):
  * pergunta vazia/longa demais, dados pessoais (CPF, telefone, e-mail) -> bloqueio;
  * pedido de diagnóstico -> modo ressalva (instrução extra + verificação na saída);
  * fontes que não cobrem a pergunta -> recusa SEM chamar o modelo.
Depois do modelo (o que o sistema obriga):
  * citação [n] tem de apontar para um trecho que entrou na janela;
  * código BNCC citado tem de existir literalmente em um trecho recuperado;
  * em modo ressalva, a resposta tem de declarar que não faz diagnóstico.
Depois do modelo, nos modos da tabela de decisão (prompt v2.0):
  * "Fundamentado nas referências: Sim / Em parte" tem de citar uma REFERÊNCIA (documento oficial ou produção
    científica); citando só relato ou nada, a linha vira "Não encontrado nas referências";
  * no modo "experiencia", a resposta tem de avisar que a orientação não foi encontrada nas referências;
  * no modo "referencia+experiencia", relato citado fora da seção "Relatos de experiência" gera alerta.
Acrescentadas na v2.1 (só somam verificações; nenhuma regra acima foi relaxada):
  * pergunta claramente fora do escopo (série 4º–9º ano ou matéria de outra área, sem tema de alfabetização) -> recusa
    SEM chamar o modelo, no mesmo caminho da falta de cobertura;
  * código BNCC malformado (EF02LP4) passa a ser capturado e removido se não constar literalmente nas fontes;
  * alerta quando a resposta passa do limite de palavras pedido no prompt;
  * alerta para citação autor-ano (Silva, 2020) cujo autor não aparece em nenhuma fonte da janela.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .config import LIMIAR_COBERTURA, MAX_PALAVRAS, MAX_PERGUNTA_CHARS, REFERENCIAS
from .indice import grupo_do_doc
from .prompts import ETIQUETA_GRUPO, PREFIXO_FUNDAMENTACAO

# Código BNCC; os quadros de Habilidades Essenciais desdobram alguns com uma letra (EF02LP01A).
# 1 a 3 dígitos no fim: "EF02LP4" (inventado e malformado) também é capturado e removido se não estiver nas fontes.
BNCC_RE = re.compile(r"\b(EF\d{2}LP\d{1,3}[A-Z]?)\b")
CITA_RE = re.compile(r"\[(\d{1,2}(?:\s*[,;]\s*\d{1,2})*)\]")

_PII = [
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "CPF"),
    (re.compile(r"\(?\b\d{2}\)?\s?9?\d{4}-?\d{4}\b"), "telefone"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "e-mail"),
]

_TRANSTORNOS = r"(dislex\w*|tdah|tea\b|autis\w*|transtorno\w*|disgraf\w*|discalcul\w*|d[ée]fici\w* de aten)"
_PEDE_DIAG = [
    re.compile(r"\b(tem|teria|é|e|seria|será|sera|possui|apresenta|indica|significa|sinal de)\b[^?.!]{0,60}\b" + _TRANSTORNOS + r"[^.!]{0,40}\?", re.I),
    re.compile(r"\b(diagnostic\w*|laudo)\b[^.!]{0,60}\?", re.I),
    re.compile(r"\b(diagnosticar|dar (o |um )?diagn[óo]stico|fechar (o |um )?diagn[óo]stico)\b", re.I),
]
_NEGA_DIAG = re.compile(
    r"(n[ãa]o (faz|fa[çc]o|realiza|realizo|pode|posso|[ée] poss[ií]vel|cabe)\w*[^.]{0,80}diagn[óo]stic"
    r"|sem (fazer|realizar) diagn[óo]stic|n[ãa]o (confirmam?|permitem?|indicam?|significam?)[^.]{0,60}" + _TRANSTORNOS
    + r"|n[ãa]o [ée] (poss[ií]vel|adequado) (concluir|afirmar|atribuir)|avalia[çc][ãa]o (profissional|especializada))",
    re.I,
)
FRASE_DIAGNOSTICO = (
    "> **Aviso do sistema:** este assistente não faz diagnóstico de dislexia, TDAH, TEA ou outros transtornos. "
    "Só uma avaliação profissional especializada pode concluir isso; abaixo há apenas orientação pedagógica."
)
FRASE_SEM_COBERTURA = (
    "Os documentos da base não trazem evidência suficiente sobre isso. "
    "O assistente cobre alfabetização em Língua Portuguesa no 2º e 3º anos (BNCC, Habilidades Essenciais, Manual ABC e artigos sobre dislexia). "
    "Reformule a pergunta dentro desse escopo ou consulte a coordenação pedagógica."
)
FRASE_SO_EXPERIENCIA = (
    "> **Aviso do sistema:** esta orientação vem de relatos de experiência e não foi encontrada nas referências "
    "da base (documentos oficiais e produção científica)."
)
# Aviso do modo "experiencia" (o prompt pede: "…e não foi encontrada nas referências da base.").
# "foi encontrad" não casa com a linha de fundamentação "Não encontrado nas referências".
_AVISO_EXP_RE = re.compile(r"n[ãa]o foi encontrad\w* nas refer[êe]ncias", re.I)
_SECAO_RELATOS_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*Relatos de experi[êe]ncia\b", re.I | re.M)
_FUND_RE = re.compile(re.escape(PREFIXO_FUNDAMENTACAO) + r"[ \t]*(Sim|Em parte|N[ãa]o encontrado nas refer[êe]ncias)([^\n*]*)", re.I)
MODOS_POR_GRUPO = ("referencia", "referencia+experiencia", "experiencia")


# Escopo: o corpus cobre alfabetização (1º ao 3º ano). Uma pergunta só é "fora do escopo" quando cita série 4º–9º ano, nível
# fora do corpus ou matéria de outra área E não fala de alfabetização. "Aluno de 5º ano que ainda não lê" continua dentro.
# Série só conta com indicador ordinal ("5º ano", "quinto ano"): "criança de 7 anos" é idade, não série.
_ORDINAIS = {"primeiro": 1, "segundo": 2, "terceiro": 3, "quarto": 4, "quinto": 5, "sexto": 6, "setimo": 7, "sétimo": 7,
             "oitavo": 8, "nono": 9}
_SERIE_NUM_RE = re.compile(r"\b(\d{1,2})\s*[ºo°ª]\s*(?:ano|s[ée]rie)\b", re.I)
_SERIE_EXT_RE = re.compile(r"\b(primeiro|segundo|terceiro|quarto|quinto|sexto|s[ée]timo|oitavo|nono)\s+(?:ano|s[ée]rie)\b", re.I)
_NIVEL_FORA_RE = re.compile(r"\b(ensino m[ée]dio|anos finais|ensino fundamental ii|ensino superior|faculdade)\b", re.I)
_MATERIA_FORA_RE = re.compile(
    r"\b(fra[çc][ãa]o|fra[çc][õo]es|matem[áa]tica|equa[çc][ãa]o|equa[çc][õo]es|geometria|tabuada|qu[íi]mica|geografia|biologia)\b", re.I)
_ALFABETIZACAO_RE = re.compile(
    r"alfabetiz|letr|s[íi]laba|fon[êe]m|fonol[óo]g|\bler\b|\bl[êe]\b|\blei\w*|leitur|escrev|escrit|palavra|\brima|dislex|disgraf"
    r"|ortograf|grafema|decodific", re.I)


def fora_do_escopo(pergunta: str) -> str:
    """Motivo (texto curto) se a pergunta está claramente fora do escopo do corpus; "" se não."""
    series = {int(m) for m in _SERIE_NUM_RE.findall(pergunta)}
    series |= {_ORDINAIS[m.lower()] for m in _SERIE_EXT_RE.findall(pergunta)}
    if any(s <= 3 for s in series) or _ALFABETIZACAO_RE.search(pergunta):
        return ""  # cita 1º–3º ano ou fala de alfabetização: segue para a busca (o limiar de cobertura decide)
    fora_serie = sorted(s for s in series if 4 <= s <= 9)
    if fora_serie:
        return f"série fora do escopo ({fora_serie[0]}º ano)"
    if (m := _NIVEL_FORA_RE.search(pergunta)):
        return f"nível fora do escopo ({m.group(1).lower()})"
    if (m := _MATERIA_FORA_RE.search(pergunta)):
        return f"matéria fora do escopo ({m.group(1).lower()})"
    return ""


@dataclass
class Entrada:
    ok: bool
    motivo: str = ""
    mensagem: str = ""
    pede_diagnostico: bool = False
    fora_escopo: str = ""  # motivo, quando a pergunta está claramente fora do escopo ("" = dentro)


def checar_entrada(pergunta: str) -> Entrada:
    p = pergunta.strip()
    if len(p) < 8:
        return Entrada(False, "pergunta_vazia", "Escreva a pergunta com um pouco mais de contexto (ano, dificuldade observada).")
    if len(p) > MAX_PERGUNTA_CHARS:
        return Entrada(False, "pergunta_longa", f"A pergunta passou de {MAX_PERGUNTA_CHARS} caracteres. Resuma o caso.")
    for rx, nome in _PII:
        if rx.search(p):
            return Entrada(False, "dado_pessoal", f"A pergunta parece conter dado pessoal ({nome}). Remova-o: o assistente não precisa identificar o aluno (LGPD).")
    pede = any(rx.search(p) for rx in _PEDE_DIAG)
    return Entrada(True, pede_diagnostico=pede, fora_escopo=fora_do_escopo(p))


@dataclass
class Cobertura:
    coberta: bool
    melhor_cos: float
    limiar: float = LIMIAR_COBERTURA


def checar_cobertura(candidatos: list[dict]) -> Cobertura:
    melhor = max((c["cos"] for c in candidatos), default=0.0)
    return Cobertura(melhor >= LIMIAR_COBERTURA, round(melhor, 4))


@dataclass
class Verificacao:
    citacoes_validas: list[int] = field(default_factory=list)
    citacoes_invalidas: list[int] = field(default_factory=list)
    codigos_ok: list[str] = field(default_factory=list)
    codigos_removidos: list[str] = field(default_factory=list)
    sem_citacao: bool = False
    modo_ressalva: bool = False
    ressalva_forcada: bool = False
    truncada: bool = False
    # Modos da tabela de decisão (prompt v2.0)
    modo: str | None = None
    fundamentacao_corrigida: int = 0           # linhas "Sim / Em parte" sem referência, reescritas pelo sistema
    fundamentacao_tipo_divergente: int = 0     # tipo escrito (documento oficial / produção científica) ≠ fonte citada
    aviso_experiencia_forcado: bool = False
    relatos_fora_da_secao: list[int] = field(default_factory=list)
    secao_relatos_ausente: bool = False
    # v2.1 (só alertas)
    palavras: int = 0                          # palavras da resposta antes da seção Fontes
    excesso_palavras: bool = False             # passou de MAX_PALAVRAS (limite pedido no prompt)
    autores_fora_das_fontes: list[str] = field(default_factory=list)  # "Silva (2020)" sem o autor em nenhuma fonte da janela
    veredito: str = "ok"
    alertas: list[str] = field(default_factory=list)

    def dict(self) -> dict:
        return asdict(self)


def _numeros(texto: str) -> list[int]:
    return [int(n) for g in CITA_RE.findall(texto) for n in re.split(r"\s*[,;]\s*", g)]


def _verificar_grupos(texto: str, trechos: list[dict], modo: str, v: Verificacao) -> str:
    """Regras do prompt v2.0. Roda depois da limpeza de citações: todo [n] restante existe na janela."""
    grupo = {i: grupo_do_doc(c["doc"]) for i, c in enumerate(trechos, 1)}
    relatos = {n for n, g in grupo.items() if g == "experiencia"}

    # 4) Linha de fundamentação: "Sim" / "Em parte" só valem citando uma referência.
    def _fund(m: re.Match) -> str:
        rotulo, resto = m.group(1), m.group(2)
        if rotulo.lower().startswith("n"):  # "Não encontrado nas referências": nada a conferir
            return m.group(0)
        refs = [n for n in _numeros(resto) if grupo.get(n) in REFERENCIAS]
        if not refs:
            v.fundamentacao_corrigida += 1
            return f"{PREFIXO_FUNDAMENTACAO} Não encontrado nas referências"
        escritos = {g for g, rot in ETIQUETA_GRUPO.items() if rot in resto.lower()}
        if escritos and escritos != {grupo[n] for n in refs}:
            v.fundamentacao_tipo_divergente += 1
        return m.group(0)

    texto = _FUND_RE.sub(_fund, texto)
    if v.fundamentacao_corrigida:
        v.alertas.append(f"Fundamentação sem referência citada reescrita como \"Não encontrado nas referências\": {v.fundamentacao_corrigida} linha(s)")
    if v.fundamentacao_tipo_divergente:
        v.alertas.append(f"Tipo de referência na fundamentação não confere com a fonte citada: {v.fundamentacao_tipo_divergente} linha(s)")

    secao = _SECAO_RELATOS_RE.search(texto)
    # 5) Só relatos: a resposta tem de avisar que não há base nas referências; se não avisar, o sistema avisa.
    if modo == "experiencia":
        inicio = texto[: secao.start()] if secao else texto[:400]
        if not _AVISO_EXP_RE.search(inicio):
            texto = FRASE_SO_EXPERIENCIA + "\n\n" + texto
            v.aviso_experiencia_forcado = True
            v.alertas.append("O modelo não avisou que a orientação vem só de relatos; o sistema inseriu o aviso.")

    # 6) Misto: Orientação, Atividade e Acompanhamento se baseiam só nas referências.
    if modo == "referencia+experiencia" and relatos:
        antes = texto[: secao.start()] if secao else texto
        v.relatos_fora_da_secao = sorted({n for n in _numeros(antes) if n in relatos})
        if v.relatos_fora_da_secao:
            v.alertas.append(f"Relato citado fora da seção \"Relatos de experiência\": {v.relatos_fora_da_secao}")
        if not secao:
            v.secao_relatos_ausente = True
            v.alertas.append("A seção \"Relatos de experiência\" não aparece na resposta.")
    return texto


_FONTES_RE = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*)?\**[ \t]*Fontes\b", re.I | re.M)
_MAIUSC, _MINUSC = "A-ZÁÉÍÓÚÂÊÔÃÕÇ", "a-záéíóúâêôãõç"
# "Silva (2020)", "Hall et al. (2020)", "(Ramli, 2021)", "Hall e Silva, 2019": captura o primeiro sobrenome e o ano.
_AUTOR_ANO_RE = re.compile(
    rf"\b([{_MAIUSC}][{_MINUSC}]{{2,}})(?:\s+et\s+al\.?|\s+e\s+[{_MAIUSC}][{_MINUSC}]+)?\s*[,(]\s*((?:19|20)\d{{2}})\b")


def _corpo(texto: str) -> str:
    """A resposta sem a lista de fontes (que pode trazer autor-ano legítimo e não conta no limite de palavras)."""
    m = _FONTES_RE.search(texto)
    return texto[: m.start()] if m else texto


def _autores_fora_das_fontes(corpo: str, trechos: list[dict], rotulos: dict[str, str] | None) -> list[str]:
    """Citações autor-ano cujo autor não aparece no texto nem no rótulo de nenhum trecho da janela."""
    visivel = " ".join(c["texto"] for c in trechos).lower() + " " + " ".join(
        (rotulos or {}).get(c["doc"], c["doc"]) for c in trechos).lower()
    achados: list[str] = []
    for autor, ano in _AUTOR_ANO_RE.findall(corpo):
        if autor.lower() not in visivel and f"{autor} ({ano})" not in achados:
            achados.append(f"{autor} ({ano})")
    return achados


def verificar_saida(resposta: str, trechos: list[dict], pede_diagnostico: bool, done_reason: str | None,
                    modo: str | None = None, rotulos: dict[str, str] | None = None) -> tuple[str, Verificacao]:
    """`modo`: da tabela de decisão do pipeline. None ou "unico" (fluxo antigo): só as regras 1–3.
    `rotulos` (id do documento -> rótulo) só serve à checagem autor-ano; sem ele, vale o texto dos trechos."""
    v = Verificacao(modo_ressalva=pede_diagnostico, truncada=(done_reason == "length"), modo=modo)
    texto = resposta

    # 1) Citações [n] (ou [n, m]) precisam existir na janela.
    k = len(trechos)
    usadas = sorted({int(n) for g in CITA_RE.findall(texto) for n in re.split(r"\s*[,;]\s*", g)})
    v.citacoes_validas = [n for n in usadas if 1 <= n <= k]
    v.citacoes_invalidas = [n for n in usadas if not 1 <= n <= k]
    if v.citacoes_invalidas:
        def _filtra(m: re.Match) -> str:
            ok = [n for n in re.split(r"\s*[,;]\s*", m.group(1)) if 1 <= int(n) <= k]
            return f"[{', '.join(ok)}]" if ok else ""
        texto = CITA_RE.sub(_filtra, texto)
        v.alertas.append(f"Citação para fonte inexistente removida: {v.citacoes_invalidas}")
    if not v.citacoes_validas:
        v.sem_citacao = True
        v.alertas.append("A resposta não cita nenhuma fonte recuperada.")

    # 2) Códigos BNCC precisam aparecer literalmente em algum trecho recuperado.
    # Um código desdobrado no trecho (EF02LP01A) também valida o código BNCC de origem (EF02LP01);
    # o inverso não vale: EF02LP01A só passa se aparecer assim no trecho.
    no_contexto = set()
    for c in trechos:
        achados = BNCC_RE.findall(c["texto"])
        no_contexto.update(achados)
        no_contexto.update(x[:8] for x in achados)
    citados = sorted(set(BNCC_RE.findall(texto)))
    v.codigos_ok = [c for c in citados if c in no_contexto]
    v.codigos_removidos = [c for c in citados if c not in no_contexto]
    if v.codigos_removidos:
        texto = BNCC_RE.sub(lambda m: m.group(1) if m.group(1) in no_contexto else "[código BNCC removido pelo sistema: não consta nas fontes]", texto)
        v.alertas.append(f"Código(s) BNCC fora das fontes removido(s): {', '.join(v.codigos_removidos)}")

    if modo in MODOS_POR_GRUPO:
        texto = _verificar_grupos(texto, trechos, modo, v)

    # 2b) Só alertas (v2.1): tamanho e autor-ano. Não alteram o texto.
    corpo = _corpo(texto)
    v.palavras = len(corpo.split())
    if v.palavras > MAX_PALAVRAS:
        v.excesso_palavras = True
        v.alertas.append(f"Resposta com {v.palavras} palavras; o prompt pede no máximo {MAX_PALAVRAS}.")
    v.autores_fora_das_fontes = _autores_fora_das_fontes(corpo, trechos, rotulos)
    if v.autores_fora_das_fontes:
        v.alertas.append(f"Citação autor-ano sem o autor em nenhuma fonte da janela: {', '.join(v.autores_fora_das_fontes)}")

    # 3) Modo ressalva: a resposta tem de negar o diagnóstico; se não negar, o sistema força o aviso.
    # Fica por último para o aviso de diagnóstico ser sempre a primeira linha.
    if pede_diagnostico and not _NEGA_DIAG.search(texto):
        texto = FRASE_DIAGNOSTICO + "\n\n" + texto
        v.ressalva_forcada = True
        v.alertas.append("O modelo não declarou o limite de diagnóstico; o sistema inseriu o aviso.")

    if v.truncada:
        v.alertas.append("Resposta cortada pelo limite de tokens.")

    v.veredito = "ok" if not v.alertas else "alertas"
    return texto, v
