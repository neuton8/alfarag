"""Memória de casos: anotações do professor sobre um aluno, identificado só por apelido, guardadas em SQLite.

Desligada por padrão (MEMORIA=0). Quando ligada, o professor escreve em linguagem natural e o sistema entende por regras,
sem chamar o modelo:
    "Anote sobre o Juninho: troca b e d ao escrever"      -> grava a nota
    "O que você anotou sobre o Juninho?"                    -> devolve as notas, literais
    "Esqueça o Juninho" / "Esqueça tudo"                    -> apaga
E, quando uma pergunta cita um apelido já salvo ("o Juninho está mal, o que fazer?"), as notas entram no prompt numa
FICHA DO CASO separada das FONTES: é contexto do professor, não evidência, e não se cita com [n].

Por que sem LLM na escrita: um modelo de 2B decidindo o que lembrar gravaria o que o professor não pediu.

Privacidade (leia antes de ligar):
  * só apelido ou primeiro nome; o harness bloqueia CPF, telefone e e-mail também na nota, antes de gravar;
  * cada nota pertence a uma conta (cascata ao excluir a conta) ou, em servidor local sem login, ao dono "local";
    em servidor aberto sem conta (ex.: Hugging Face Space, disco efêmero) a memória fica desligada;
  * só vai ao modelo local: com modelo remoto a ficha não é enviada;
  * o texto das notas nunca vai para o traço (só contagens) e as notas expiram em MEMORIA_DIAS.
"""
from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import contas, harness
from .config import MEMORIA_DB, MEMORIA_DIAS, MEMORIA_MAX_CASOS, MEMORIA_MAX_NOTAS, MEMORIA_NOTA_MAX

NOTA_MIN = 5
FICHA_NOTAS = 8       # notas mais recentes de cada caso que entram no prompt
FICHA_CASOS = 2       # casos citados na mesma pergunta
FICHA_CHARS = 1200    # teto do bloco, para a ficha não competir com as FONTES na janela
LOCAL = "local"       # dono da memória em servidor local sem login

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS memoria_casos (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    usuario   TEXT NOT NULL{fk},
    apelido   TEXT NOT NULL COLLATE NOCASE,
    criado_em TEXT NOT NULL,
    UNIQUE (usuario, apelido)
);
CREATE TABLE IF NOT EXISTS memoria_notas (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    caso_id   INTEGER NOT NULL REFERENCES memoria_casos(id) ON DELETE CASCADE,
    texto     TEXT NOT NULL,
    criada_em TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memoria_notas_caso ON memoria_notas(caso_id);
"""


class ErroMemoria(Exception):
    """Mensagem segura para mostrar ao professor."""


# ---------- Banco ----------
_PRONTOS: set[str] = set()


def _caminho() -> str:
    """Com contas (CONTAS_DB), as tabelas ficam no mesmo arquivo e caem em cascata junto com a conta."""
    return contas.CONTAS_DB or MEMORIA_DB


def _abrir() -> sqlite3.Connection:
    cam = _caminho()
    caminho = Path(cam)
    if cam not in _PRONTOS:
        caminho.parent.mkdir(parents=True, exist_ok=True)
        os.close(os.open(caminho, os.O_CREAT | os.O_RDWR, 0o600))  # nasce 600; o SQLite copia a permissão no -wal
        os.chmod(caminho, 0o600)
        con = sqlite3.connect(caminho, isolation_level=None)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_ESQUEMA.format(fk=" REFERENCES contas(usuario) ON DELETE CASCADE" if contas.CONTAS_DB else ""))
        finally:
            con.close()
        _PRONTOS.add(cam)
    con = sqlite3.connect(caminho, timeout=10, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextmanager
def _db(escrita: bool = False):
    con = _abrir()
    try:
        con.execute("BEGIN IMMEDIATE" if escrita else "BEGIN")
        yield con
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _agora() -> datetime:
    return datetime.now(timezone.utc)


def _purgar(con: sqlite3.Connection) -> None:
    """Retenção: notas mais velhas que MEMORIA_DIAS somem, e casos sem nota também."""
    limite = (_agora() - timedelta(days=MEMORIA_DIAS)).isoformat(timespec="seconds")
    con.execute("DELETE FROM memoria_notas WHERE criada_em < ?", (limite,))
    con.execute("DELETE FROM memoria_casos WHERE id NOT IN (SELECT DISTINCT caso_id FROM memoria_notas)")


# ---------- Operações ----------
def _apelido_valido(apelido: str) -> str:
    a = (apelido or "").strip()
    if not _NOME_RE.fullmatch(a) or a in _NAO_E_NOME:
        raise ErroMemoria("Use só o primeiro nome ou um apelido do aluno, com a inicial maiúscula (por exemplo, “Juninho”).")
    return a


def salvar_nota(dono: str, apelido: str, texto: str) -> tuple[str, int]:
    """Grava uma nota no caso (cria o caso se for novo). Devolve (apelido, total de notas do caso)."""
    apelido = _apelido_valido(apelido)
    texto = re.sub(r"\s+", " ", texto or "").strip()
    if len(texto) < NOTA_MIN:
        raise ErroMemoria("A nota ficou curta demais. Escreva o que observou, por exemplo: “Anote sobre o Juninho: troca b e d”.")
    if len(texto) > MEMORIA_NOTA_MAX:
        raise ErroMemoria(f"A nota passou de {MEMORIA_NOTA_MAX} caracteres. Resuma e tente de novo.")
    pii = harness.dado_pessoal(texto)  # o mesmo bloqueio de CPF, telefone e e-mail da pergunta
    if pii:
        raise ErroMemoria(f"A nota parece conter dado pessoal ({pii}). Remova-o: guarde só o primeiro nome ou apelido do aluno (LGPD).")
    with _db(escrita=True) as con:
        _purgar(con)
        caso = con.execute("SELECT id FROM memoria_casos WHERE usuario = ? AND apelido = ?", (dono, apelido)).fetchone()
        if caso is None:
            if con.execute("SELECT COUNT(*) FROM memoria_casos WHERE usuario = ?", (dono,)).fetchone()[0] >= MEMORIA_MAX_CASOS:
                raise ErroMemoria(f"Há {MEMORIA_MAX_CASOS} casos salvos, que é o limite. Apague algum (“esqueça o <nome>”) para criar outro.")
            cur = con.execute("INSERT INTO memoria_casos (usuario, apelido, criado_em) VALUES (?, ?, ?)", (dono, apelido, _agora().isoformat(timespec="seconds")))
            caso_id = cur.lastrowid
        else:
            caso_id = caso["id"]
        if con.execute("SELECT COUNT(*) FROM memoria_notas WHERE caso_id = ?", (caso_id,)).fetchone()[0] >= MEMORIA_MAX_NOTAS:
            raise ErroMemoria(f"O caso já tem {MEMORIA_MAX_NOTAS} notas, que é o limite. Apague o caso e recomece com um resumo.")
        con.execute("INSERT INTO memoria_notas (caso_id, texto, criada_em) VALUES (?, ?, ?)", (caso_id, texto, _agora().isoformat(timespec="seconds")))
        n = con.execute("SELECT COUNT(*) FROM memoria_notas WHERE caso_id = ?", (caso_id,)).fetchone()[0]
        nome = con.execute("SELECT apelido FROM memoria_casos WHERE id = ?", (caso_id,)).fetchone()["apelido"]
    return nome, n


def listar(dono: str) -> list[dict]:
    """Todos os casos do dono, cada um com as notas em ordem cronológica."""
    with _db(escrita=True) as con:  # escrita porque também expurga o que venceu
        _purgar(con)
        casos = con.execute("SELECT id, apelido, criado_em FROM memoria_casos WHERE usuario = ? ORDER BY apelido", (dono,)).fetchall()
        saida = []
        for c in casos:
            notas = con.execute("SELECT id, texto, criada_em FROM memoria_notas WHERE caso_id = ? ORDER BY criada_em, id", (c["id"],)).fetchall()
            saida.append({"id": c["id"], "apelido": c["apelido"], "notas": [{"id": n["id"], "texto": n["texto"], "criada_em": n["criada_em"]} for n in notas]})
    return saida


def esquecer_caso(dono: str, apelido: str | None = None, caso_id: int | None = None) -> int:
    """Apaga um caso (por apelido ou id) com as notas. Devolve quantas notas foram apagadas."""
    with _db(escrita=True) as con:
        if caso_id is not None:
            c = con.execute("SELECT id FROM memoria_casos WHERE usuario = ? AND id = ?", (dono, caso_id)).fetchone()
        else:
            c = con.execute("SELECT id FROM memoria_casos WHERE usuario = ? AND apelido = ?", (dono, apelido or "")).fetchone()
        if c is None:
            return 0
        n = con.execute("SELECT COUNT(*) FROM memoria_notas WHERE caso_id = ?", (c["id"],)).fetchone()[0]
        con.execute("DELETE FROM memoria_casos WHERE id = ?", (c["id"],))
    return n


def esquecer_tudo(dono: str) -> int:
    with _db(escrita=True) as con:
        n = con.execute("SELECT COUNT(*) FROM memoria_notas WHERE caso_id IN (SELECT id FROM memoria_casos WHERE usuario = ?)", (dono,)).fetchone()[0]
        con.execute("DELETE FROM memoria_casos WHERE usuario = ?", (dono,))
    return n


# ---------- Comandos em linguagem natural ----------
_NOME = r"[A-ZÀ-Ý][a-zà-ÿ'’-]{1,29}"  # nome ou apelido: uma palavra com inicial maiúscula (não pega "aluno", "criança"…)
_NOME_RE = re.compile(_NOME)
# Palavras que começam frase e parecem nome ("Em", "Meu"…): nunca viram apelido.
_NAO_E_NOME = {"Em", "No", "Na", "Nos", "Nas", "Meu", "Minha", "Quando", "Como", "Se", "Para", "Que", "Hoje", "Ontem", "Todo", "Toda",
               "Ele", "Ela", "Eles", "Elas", "Aluno", "Aluna", "Crianca", "Criança", "Sobre", "Isso", "Isto", "Nao", "Não", "Por", "Com", "Sem"}
_ART = r"(?i:(?:(?:o|a|do|da|ao|à)\s+)?(?:(?:aluno|aluna|crian[çc]a|menino|menina)\s+)?)"
_SALVAR = re.compile(
    r"^\s*(?i:(?:por favor[, ]+)?(?:anote|anota|registre|registra|lembre|lembra|guarde|guarda|salve|salva)\b(?:-se)?)"
    r"(?i:(?:\s+que|\s+sobre|\s+isto:|\s+isso:)?)\s*:?\s*" + _ART + rf"(?P<apelido>{_NOME})\b\s*[:,;–—-]?\s*(?P<nota>\S.{{4,}})$", re.S)
_RECORDAR = [
    re.compile(r"^\s*(?i:o\s+que)\s+(?i:(?:eu\s+)?(?:te\s+|lhe\s+)?(?:falei|disse|contei|anotei|registrei|salvei|guardei)"
               r"|(?:voc[êe]|vc|tu)\s+(?:anotou|registrou|guardou|salvou|tem|sabe))"
               r"(?i:(?:\s+(?:antes|ontem|sobre|d[oa]|a\s+respeito\s+d[oa]|salvo|anotado))*)\s*" + _ART + rf"(?P<apelido>{_NOME})?\s*\??\s*$"),
    re.compile(r"^\s*(?i:o\s+que\s+ficou\s+(?:salvo|anotado|registrado|guardado))(?i:(?:\s+(?:d[oa]|sobre))?)\s*" + _ART + rf"(?P<apelido>{_NOME})?\s*\??\s*$"),
    re.compile(r"^\s*(?i:(?:voc[êe]\s+)?lembra\s+(?:d[oa]|dele|dela))\s*" + rf"(?P<apelido>{_NOME})?\s*\??\s*$"),
]
_ESQ = r"(?i:esque[çc]a|apague|apaga|remova|remove|exclua|limpe)"
_ESQUECER_TUDO = re.compile(rf"^\s*{_ESQ}\s+(?i:tudo|todas\s+as\s+notas|toda\s+a\s+mem[óo]ria|a\s+mem[óo]ria)\s*[.!]?\s*$")
_ESQUECER = re.compile(rf"^\s*{_ESQ}\s+(?i:(?:tudo\s+)?(?:sobre|d[oa]|as\s+notas\s+d[oa]|a\s+ficha\s+d[oa]|o\s+caso\s+d[oa])\s+)?" + _ART + rf"(?P<apelido>{_NOME})\s*[.!]?\s*$")


@dataclass
class Comando:
    tipo: str            # "salvar" | "recordar" | "esquecer" | "esquecer_tudo"
    apelido: str | None = None
    nota: str | None = None


def interpretar(texto: str) -> Comando | None:
    """Reconhece só comandos claros. Pergunta normal (ou frase ambígua) devolve None e segue para o fluxo de sempre."""
    t = (texto or "").strip()
    if not t or len(t) > 1200:
        return None
    if _ESQUECER_TUDO.match(t):
        return Comando("esquecer_tudo")
    for rx in _RECORDAR:
        m = rx.match(t)
        if m:
            ap = m.group("apelido")
            return Comando("recordar", ap if ap and ap not in _NAO_E_NOME else None)
    m = _ESQUECER.match(t)
    if m and m.group("apelido") not in _NAO_E_NOME:
        return Comando("esquecer", m.group("apelido"))
    m = _SALVAR.match(t)
    if m and m.group("apelido") not in _NAO_E_NOME:
        return Comando("salvar", m.group("apelido"), m.group("nota").strip())
    return None


def _data(iso: str) -> str:
    return datetime.fromisoformat(iso).strftime("%d/%m")


def executar(dono: str, cmd: Comando) -> str:
    """Executa o comando e devolve o texto (markdown) da resposta. Nunca chama o modelo."""
    try:
        if cmd.tipo == "salvar":
            nome, n = salvar_nota(dono, cmd.apelido or "", cmd.nota or "")
            return (f"Anotei sobre **{nome}** ({n} nota{'s' if n > 1 else ''} no caso).\n\n"
                    f"Quando você perguntar algo citando {nome}, eu uso essa ficha como contexto. "
                    f"Para ver: “o que você anotou sobre o {nome}?”. Para apagar: “esqueça o {nome}”. "
                    "Use só o primeiro nome ou apelido, nunca CPF, telefone ou sobrenome.")
        if cmd.tipo == "recordar":
            casos = [c for c in listar(dono) if cmd.apelido is None or c["apelido"].lower() == cmd.apelido.lower()]
            if not casos:
                return (f"Não há nada salvo sobre **{cmd.apelido}**." if cmd.apelido else "Ainda não há nada salvo.") + \
                    " Para guardar algo: “Anote sobre o Juninho: troca b e d ao escrever”."
            linhas = []
            for c in casos:
                linhas.append(f"**{c['apelido']}** ({len(c['notas'])} nota{'s' if len(c['notas']) != 1 else ''})")
                linhas += [f"- {_data(n['criada_em'])}: {n['texto']}" for n in c["notas"]]
            return "\n".join(linhas) + "\n\nSão as suas anotações, ditas por você; não vêm dos documentos da base."
        if cmd.tipo == "esquecer":
            n = esquecer_caso(dono, apelido=cmd.apelido)
            return f"Apaguei {n} nota{'s' if n != 1 else ''} sobre **{cmd.apelido}**." if n else f"Não havia nada salvo sobre **{cmd.apelido}**."
        if cmd.tipo == "esquecer_tudo":
            n = esquecer_tudo(dono)
            return f"Apaguei toda a memória ({n} nota{'s' if n != 1 else ''})." if n else "A memória já estava vazia."
    except ErroMemoria as e:
        return str(e)
    raise ValueError(f"comando desconhecido: {cmd.tipo}")


# ---------- Ficha para o prompt ----------
def ficha_para(dono: str, pergunta: str) -> dict | None:
    """Casos do dono cujo apelido aparece na pergunta. Devolve {bloco, busca, casos, notas} ou None.
    `bloco` vai ao prompt (separado das FONTES); `busca` enriquece a consulta, para "o Juninho está mal?" não cair em
    sem_cobertura por falar só do nome; o texto das notas nunca vai ao traço (só `casos` e `notas`)."""
    achados = []
    for c in listar(dono):
        if c["notas"] and re.search(rf"(?<![\wÀ-ÿ]){re.escape(c['apelido'])}(?![\wÀ-ÿ])", pergunta, re.I):
            achados.append(c)
    if not achados:
        return None
    achados = achados[:FICHA_CASOS]
    linhas, busca, n_notas = [], [], 0
    for c in achados:
        linhas.append(f"{c['apelido']}:")
        for n in c["notas"][-FICHA_NOTAS:]:
            linhas.append(f"- {_data(n['criada_em'])}: {n['texto']}")
            busca.append(n["texto"])
            n_notas += 1
    corpo = "\n".join(linhas)[:FICHA_CHARS]
    bloco = ("FICHA DO CASO (anotações do próprio professor; é contexto, não é evidência e não se cita com [n]; "
             "a orientação continua baseada só nas FONTES):\n" + corpo)
    return {"bloco": bloco, "busca": " ".join(busca)[:400], "casos": len(achados), "notas": n_notas}
