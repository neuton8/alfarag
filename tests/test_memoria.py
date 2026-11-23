"""Memória de casos: comandos em linguagem natural, SQLite por dono, ficha no prompt e privacidade. Tudo offline."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app import contas, main, memoria, pipeline
from .test_estruturado import FIM, Ollama, _json_do_esquema, _ref, _relato
from .test_login import ADMIN, ORIGEM, PROF, SENHA_ADMIN, SENHA_PROF, cliente, entrar, login, prof_ativa  # noqa: F401  (fixture `login`)

NOTA = "troca b e d ao escrever"
LOCAL = {"Origin": "http://testserver", "Content-Type": "application/json"}


@pytest.fixture
def mem(tmp_path, monkeypatch):
    """Memória ligada, sem contas: servidor local, banco próprio em tmp_path."""
    monkeypatch.setattr(memoria, "MEMORIA_DB", str(tmp_path / "memoria.db"))
    monkeypatch.setattr(main, "MEMORIA", True)
    monkeypatch.setattr(main, "modelos_disponiveis", lambda: [])
    return tmp_path / "memoria.db"


@pytest.fixture
def local():
    return TestClient(main.app, client=("127.0.0.1", 50000))


def _ndjson(resp):
    return [json.loads(linha) for linha in resp.text.splitlines() if linha.strip()]


# ---------- interpretar: só comandos claros ----------
@pytest.mark.parametrize("texto,tipo,apelido,nota", [
    ("Anote sobre o Juninho: troca b e d ao escrever", "salvar", "Juninho", "troca b e d ao escrever"),
    ("anote que o Juninho troca b e d ao escrever", "salvar", "Juninho", "troca b e d ao escrever"),
    ("Lembre que o Juninho tem dificuldade com sílabas complexas", "salvar", "Juninho", "tem dificuldade com sílabas complexas"),
    ("Registre: Maria não reconhece a letra F", "salvar", "Maria", "não reconhece a letra F"),
    ("Guarde que a aluna Bia lê palavras de duas sílabas", "salvar", "Bia", "lê palavras de duas sílabas"),
    ("Por favor, anote sobre a Maria - copia da lousa sem entender", "salvar", "Maria", "copia da lousa sem entender"),
    ("o que eu falei antes?", "recordar", None, None),
    ("O que você anotou sobre o Juninho?", "recordar", "Juninho", None),
    ("o que eu te falei sobre a Maria", "recordar", "Maria", None),
    ("O que ficou salvo do Juninho?", "recordar", "Juninho", None),
    ("Você lembra do Juninho?", "recordar", "Juninho", None),
    ("Esqueça o Juninho", "esquecer", "Juninho", None),
    ("apague tudo sobre a Maria", "esquecer", "Maria", None),
    ("Esqueça tudo", "esquecer_tudo", None, None),
    ("apague a memória", "esquecer_tudo", None, None),
])
def test_interpretar_comandos(texto, tipo, apelido, nota):
    c = memoria.interpretar(texto)
    assert c is not None and (c.tipo, c.apelido, c.nota) == (tipo, apelido, nota)


@pytest.mark.parametrize("texto", [
    "O Juninho está mal, o que fazer?",                      # pergunta que cita o aluno: vai ao modelo com a ficha
    "Como ensinar um aluno do 2º ano a escrever palavras?",
    "Anote que o aluno do 2º ano troca letras",              # sem nome próprio: nada a guardar
    "Lembre que Em breve teremos prova de leitura",          # "Em" não é apelido
    "Como lembrar o alfabeto com a turma?",
    "O que fazer quando o aluno troca letras?",
    "Registre isto",
    "Esqueça",
    "Anote sobre o Juninho",                                 # sem nota
    "",
])
def test_pergunta_comum_nao_vira_comando(texto):
    assert memoria.interpretar(texto) is None


# ---------- notas, casos e dono ----------
def test_salvar_listar_e_esquecer(mem):
    assert memoria.salvar_nota("local", "Juninho", NOTA) == ("Juninho", 1)
    assert memoria.salvar_nota("local", "Juninho", "não reconhece o F") == ("Juninho", 2)  # mesmo caso, com a segunda nota
    casos = memoria.listar("local")
    assert [c["apelido"] for c in casos] == ["Juninho"] and [n["texto"] for n in casos[0]["notas"]] == [NOTA, "não reconhece o F"]
    assert memoria.esquecer_caso("local", apelido="JUNINHO") == 2 and memoria.listar("local") == []
    assert mem.stat().st_size > 0


def test_cada_dono_so_ve_o_que_e_seu(mem):
    memoria.salvar_nota("ana", "Juninho", "nota da Ana")
    memoria.salvar_nota("bia", "Juninho", "nota da Bia")
    assert [n["texto"] for n in memoria.listar("ana")[0]["notas"]] == ["nota da Ana"]
    assert [n["texto"] for n in memoria.listar("bia")[0]["notas"]] == ["nota da Bia"]
    assert memoria.esquecer_caso("ana", apelido="Juninho") == 1 and memoria.listar("bia")  # apagar o seu não toca o dela
    assert memoria.esquecer_caso("ana", caso_id=memoria.listar("bia")[0]["id"]) == 0       # nem pelo id
    memoria.salvar_nota("ana", "Maria", "outra nota")
    assert memoria.esquecer_tudo("ana") == 1 and len(memoria.listar("bia")) == 1


@pytest.mark.parametrize("apelido", ["juninho", "Juninho Silva", "aluno", "Em", "J", "Maria123", ""])
def test_apelido_invalido_e_recusado(mem, apelido):
    with pytest.raises(memoria.ErroMemoria):
        memoria.salvar_nota("local", apelido, NOTA)
    assert memoria.listar("local") == []


@pytest.mark.parametrize("nota", ["", "oi", "x" * 600])
def test_nota_vazia_curta_ou_longa_e_recusada(mem, nota):
    with pytest.raises(memoria.ErroMemoria):
        memoria.salvar_nota("local", "Juninho", nota)


@pytest.mark.parametrize("nota", ["CPF 123.456.789-09 e troca letras", "mãe: (12) 91234-5678, troca letras", "falar com mae@exemplo.com sobre o caso"])
def test_dado_pessoal_na_nota_e_barrado_antes_de_gravar(mem, nota):
    with pytest.raises(memoria.ErroMemoria, match="dado pessoal"):
        memoria.salvar_nota("local", "Juninho", nota)
    assert memoria.listar("local") == []


def test_limites_de_casos_e_de_notas(mem, monkeypatch):
    monkeypatch.setattr(memoria, "MEMORIA_MAX_CASOS", 2)
    monkeypatch.setattr(memoria, "MEMORIA_MAX_NOTAS", 2)
    memoria.salvar_nota("local", "Ana", "primeira nota")
    memoria.salvar_nota("local", "Ana", "segunda nota")
    with pytest.raises(memoria.ErroMemoria, match="limite"):
        memoria.salvar_nota("local", "Ana", "terceira nota")
    memoria.salvar_nota("local", "Bia", "outra nota")
    with pytest.raises(memoria.ErroMemoria, match="limite"):
        memoria.salvar_nota("local", "Cris", "nota demais")


def test_notas_expiram_pela_retencao(mem, monkeypatch):
    memoria.salvar_nota("local", "Juninho", NOTA)
    con = sqlite3.connect(mem)
    con.execute("UPDATE memoria_notas SET criada_em = '2000-01-01T00:00:00+00:00'")
    con.commit()
    con.close()
    assert memoria.listar("local") == []  # nota vencida some, e o caso sem notas também
    con = sqlite3.connect(mem)
    assert con.execute("SELECT COUNT(*) FROM memoria_casos").fetchone()[0] == 0
    con.close()


def test_arquivo_do_banco_nasce_600_no_posix(mem):
    memoria.salvar_nota("local", "Juninho", NOTA)
    if hasattr(__import__("os"), "geteuid"):  # no Windows o chmod não aplica permissões POSIX
        assert mem.stat().st_mode & 0o777 == 0o600


# ---------- executar: o texto que o professor lê ----------
def test_executar_fluxo_completo(mem):
    r = memoria.executar("local", memoria.interpretar("Anote sobre o Juninho: " + NOTA))
    assert "Anotei sobre **Juninho**" in r and "nunca CPF" in r
    r = memoria.executar("local", memoria.interpretar("o que eu falei antes?"))
    assert "**Juninho** (1 nota)" in r and NOTA in r and "não vêm dos documentos" in r
    r = memoria.executar("local", memoria.interpretar("O que você anotou sobre a Maria?"))
    assert "Não há nada salvo sobre **Maria**" in r
    r = memoria.executar("local", memoria.interpretar("Esqueça o Juninho"))
    assert "Apaguei 1 nota sobre **Juninho**" in r and memoria.listar("local") == []
    assert memoria.executar("local", memoria.interpretar("Esqueça tudo")) == "A memória já estava vazia."


def test_executar_nao_explode_com_nota_recusada(mem):
    r = memoria.executar("local", memoria.interpretar("Anote sobre o Juninho: CPF 123.456.789-09 troca letras"))
    assert "dado pessoal" in r and memoria.listar("local") == []


# ---------- ficha para o prompt ----------
def test_ficha_so_quando_a_pergunta_cita_o_apelido(mem):
    memoria.salvar_nota("local", "Juninho", NOTA)
    memoria.salvar_nota("local", "Juninho", "não reconhece o F")
    memoria.salvar_nota("local", "Maria", "lê palavras de duas sílabas")
    f = memoria.ficha_para("local", "O Juninho está mal, o que fazer?")
    assert f["casos"] == 1 and f["notas"] == 2 and NOTA in f["bloco"] and "Maria" not in f["bloco"]
    assert "não é evidência" in f["bloco"] and "não se cita com [n]" in f["bloco"] and NOTA in f["busca"]
    assert memoria.ficha_para("local", "o JUNINHO não evolui, como ajudar?")["casos"] == 1  # sem diferenciar maiúscula
    assert memoria.ficha_para("local", "Como ensinar rimas?") is None
    assert memoria.ficha_para("local", "Os Juninhos da turma trocam letras") is None          # outra palavra
    assert memoria.ficha_para("outra", "O Juninho está mal?") is None                           # outro dono


# ---------- API ----------
def test_memoria_desligada_por_padrao(local, monkeypatch):
    monkeypatch.setattr(main, "modelos_disponiveis", lambda: [])
    assert local.get("/api/status").json()["memoria"] is False
    assert local.get("/api/memoria").json() == {"ativa": False, "casos": []}
    assert local.delete("/api/memoria", headers=LOCAL).status_code == 404


def test_api_comando_nao_chama_modelo_e_nao_grava_traco(mem, local, monkeypatch):
    chamadas = []
    monkeypatch.setattr(main, "responder", lambda *a, **k: chamadas.append(1) or iter(()))
    r = local.post("/api/perguntar", json={"pergunta": "Anote sobre o Juninho: " + NOTA}, headers=LOCAL)
    final = _ndjson(r)[-1]
    assert r.status_code == 200 and final["desfecho"] == "memoria" and "Anotei sobre **Juninho**" in final["resposta"]
    assert chamadas == [] and not pipeline.TRACOS_DIR.exists()  # nem modelo, nem traço com o texto da nota
    assert local.get("/api/status").json()["memoria"] is True
    lista = local.get("/api/memoria").json()
    assert lista["ativa"] and lista["casos"][0]["apelido"] == "Juninho" and lista["casos"][0]["notas"][0]["texto"] == NOTA


def test_comando_de_memoria_exige_mesma_origem(mem, local):
    """Sem cookie (servidor local), um POST simples de outra página conseguiria apagar ou inserir notas."""
    memoria.salvar_nota("local", "Juninho", NOTA)
    for fala in ("Esqueça tudo", "Esqueça o Juninho", "Anote sobre o Juninho: nota falsa de outro site"):
        corpo = json.dumps({"pergunta": fala})
        assert local.post("/api/perguntar", json={"pergunta": fala}).status_code == 403                                            # sem Origin
        assert local.post("/api/perguntar", json={"pergunta": fala}, headers={"Origin": "http://evil.example"}).status_code == 403  # outro site
        # POST simples de outro site (text/plain ou sem Content-Type): o FastAPI recente já recusa o corpo (422); versões antigas leem como JSON, e aí vale o 415
        assert local.post("/api/perguntar", content=corpo, headers={"Origin": "http://testserver", "Content-Type": "text/plain"}).status_code in (415, 422)
        assert local.post("/api/perguntar", content=corpo, headers={"Origin": "http://testserver"}).status_code in (415, 422)
    assert [n["texto"] for n in memoria.listar("local")[0]["notas"]] == [NOTA]  # nada foi apagado nem inserido
    assert local.post("/api/perguntar", json={"pergunta": "Esqueça tudo"}, headers=LOCAL).status_code == 200
    assert memoria.listar("local") == []


def test_pergunta_comum_nao_ganha_checagem_de_origem(mem, local, monkeypatch):
    """Só os comandos escrevem; quem faz pergunta (curl, scripts, a própria interface) segue como antes."""
    monkeypatch.setattr(main, "responder", lambda *a, **k: iter([{"tipo": "final", "desfecho": "respondida", "resposta": "ok"}]))
    assert local.post("/api/perguntar", json={"pergunta": "Como ensinar rimas?"}).status_code == 200
    assert local.post("/api/perguntar", json={"pergunta": "O Juninho está mal, o que fazer?"}).status_code == 200  # cita um caso, mas não grava


def test_api_pergunta_que_cita_o_caso_leva_a_ficha(mem, local, monkeypatch):
    memoria.salvar_nota("local", "Juninho", NOTA)
    vistos = {}
    monkeypatch.setattr(main, "responder", lambda p, **k: vistos.update(pergunta=p, **k) or iter([{"tipo": "final", "desfecho": "respondida", "resposta": "ok"}]))
    local.post("/api/perguntar", json={"pergunta": "O Juninho está mal, o que fazer?"})
    assert vistos["ficha"]["casos"] == 1 and NOTA in vistos["ficha"]["bloco"] and vistos["pergunta"] == "O Juninho está mal, o que fazer?"
    vistos.clear()
    local.post("/api/perguntar", json={"pergunta": "Como ensinar rimas?"})
    assert vistos["ficha"] is None


def test_api_apagar_exige_origem_e_so_apaga_o_proprio(mem, local):
    memoria.salvar_nota("local", "Juninho", NOTA)
    memoria.salvar_nota("ana", "Juninho", "da Ana")
    cid = memoria.listar("local")[0]["id"]
    assert local.delete(f"/api/memoria/{cid}").status_code == 415               # sem JSON explícito
    assert local.delete(f"/api/memoria/{cid}", headers={"Content-Type": "application/json", "Origin": "http://evil.example"}).status_code == 403
    assert local.delete(f"/api/memoria/{memoria.listar('ana')[0]['id']}", headers=LOCAL).status_code == 404  # caso de outro dono
    assert local.delete(f"/api/memoria/{cid}", headers=LOCAL).json() == {"ok": True} and memoria.listar("local") == []
    assert len(memoria.listar("ana")) == 1
    memoria.salvar_nota("local", "Maria", "outra nota")
    assert local.delete("/api/memoria", headers=LOCAL).json()["notas_apagadas"] == 1


def test_servidor_publico_sem_login_nao_tem_memoria(mem, monkeypatch):
    monkeypatch.setattr(main, "responder", lambda *a, **k: iter([{"tipo": "final", "desfecho": "respondida", "resposta": "ok"}]))
    publico = TestClient(main.app, client=("203.0.113.9", 50000))
    assert publico.get("/api/status").json()["memoria"] is False
    r = publico.post("/api/perguntar", json={"pergunta": "Anote sobre o Juninho: " + NOTA})
    assert memoria.listar("local") == [] and all(e.get("desfecho") != "memoria" for e in _ndjson(r))


# ---------- pipeline: ficha no prompt, privacidade e traço ----------
class _IndiceQueAnota:
    def __init__(self):
        self.consultas = []

    def buscar(self, consulta, docs, **kw):
        self.consultas.append(consulta)
        return [_relato(3)] if "VID" in docs else [_ref("BNCC", 3), _ref("RAMLI", 7)]


def _pipe(monkeypatch, como=None):
    indice = _IndiceQueAnota()
    ollama = Ollama(como or (lambda n, f: (_json_do_esquema(f), FIM)))
    monkeypatch.setattr(pipeline, "carregar", lambda exp: indice)
    monkeypatch.setattr(pipeline, "digest", lambda m: "abc123")
    monkeypatch.setattr(pipeline, "ESTRUTURADO", True)
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    return indice, ollama


def _ficha(mem):
    memoria.salvar_nota("local", "Juninho", NOTA)
    return memoria.ficha_para("local", "O Juninho está mal, o que fazer?")


def test_ficha_vai_ao_prompt_separada_das_fontes_e_enriquece_a_busca(mem, monkeypatch):
    indice, ollama = _pipe(monkeypatch)
    ficha = _ficha(mem)
    final = list(pipeline.responder("O Juninho está mal, o que fazer?", modelo="qwen3.5:2b-q4_K_M", ficha=ficha))[-1]
    usuario = ollama.chamadas[0]["mensagens"][1]["content"]
    assert usuario.index("FONTES:") < usuario.index("FICHA DO CASO") < usuario.index("PERGUNTA DO PROFESSOR:")
    assert NOTA in usuario and all(NOTA in c for c in indice.consultas)  # a consulta da busca leva as notas
    assert final["desfecho"] == "respondida"
    # a ficha não é fonte: não ganha número [n] nem entra na lista de fontes
    assert "FICHA" not in final["resposta"] and final["traco"]["janela"]["trechos"][0]["n"] == 1


def test_ficha_tambem_vai_no_texto_livre(mem, monkeypatch):
    _, ollama = _pipe(monkeypatch)
    monkeypatch.setattr(pipeline, "ESTRUTURADO", False)
    monkeypatch.setattr(pipeline, "chat_stream", Ollama(lambda n, f: ("**Orientação**\nRimas [1].", FIM)))
    ollama = pipeline.chat_stream
    list(pipeline.responder("O Juninho está mal, o que fazer?", modelo="qwen3.5:4b-q4_K_M", ficha=_ficha(mem)))
    assert "FICHA DO CASO" in ollama.chamadas[0]["mensagens"][1]["content"]


def test_traco_guarda_so_contagens_nunca_o_texto_da_nota(mem, monkeypatch):
    _pipe(monkeypatch)
    final = list(pipeline.responder("O Juninho está mal, o que fazer?", modelo="qwen3.5:2b-q4_K_M", ficha=_ficha(mem)))[-1]
    assert final["traco"]["memoria"] == {"casos": 1, "notas": 1}
    gravado = "".join(p.read_text(encoding="utf-8") for p in pipeline.TRACOS_DIR.glob("tracos_*.jsonl"))
    assert gravado and NOTA not in gravado and "FICHA DO CASO" not in gravado


def test_ficha_nunca_sai_para_modelo_remoto(mem, monkeypatch):
    indice, ollama = _pipe(monkeypatch, lambda n, f: ("**Orientação**\nRimas [1].", FIM))
    final = list(pipeline.responder("O Juninho está mal, o que fazer?", modelo="deepseek:deepseek-flash", ficha=_ficha(mem)))[-1]
    assert final["traco"]["memoria"] == {"desligada": "modelo remoto"}
    assert NOTA not in json.dumps(ollama.chamadas) and all(NOTA not in c for c in indice.consultas)


def test_sem_ficha_o_prompt_e_a_busca_ficam_como_antes(mem, monkeypatch):
    indice, ollama = _pipe(monkeypatch)
    final = list(pipeline.responder("Como trabalhar a consciência fonológica?", modelo="qwen3.5:2b-q4_K_M"))[-1]
    assert "FICHA" not in ollama.chamadas[0]["mensagens"][1]["content"] and "memoria" not in final["traco"]
    assert set(indice.consultas) == {"Como trabalhar a consciência fonológica?"}


# ---------- contas: isolamento e cascata ----------
def test_com_login_cada_professora_ve_so_a_propria_memoria(login, mem, monkeypatch):  # noqa: F811
    prof_ativa()
    contas.cadastrar("prof.bia", "senha-da-bia-12")
    contas.aprovar("prof.bia")
    monkeypatch.setattr(main, "responder", lambda *a, **k: iter([{"tipo": "final", "desfecho": "respondida", "resposta": "ok"}]))
    ana, bia = cliente("203.0.113.9"), cliente("198.51.100.1")
    assert entrar(ana, PROF, SENHA_PROF).status_code == 200 and entrar(bia, "prof.bia", "senha-da-bia-12").status_code == 200
    final = _ndjson(ana.post("/api/perguntar", json={"pergunta": "Anote sobre o Juninho: " + NOTA}, headers=ORIGEM))[-1]
    assert final["desfecho"] == "memoria"
    assert ana.get("/api/memoria").json()["casos"][0]["apelido"] == "Juninho"
    assert bia.get("/api/memoria").json() == {"ativa": True, "dias": memoria.MEMORIA_DIAS, "casos": []}
    assert "nada salvo" in _ndjson(bia.post("/api/perguntar", json={"pergunta": "o que eu falei antes?"}, headers=ORIGEM))[-1]["resposta"]
    assert cliente().get("/api/memoria").status_code in (303, 401)  # sem sessão não chega


def test_excluir_a_conta_apaga_a_memoria_em_cascata(login, mem):  # noqa: F811
    prof_ativa()
    memoria.salvar_nota(PROF, "Juninho", NOTA)
    memoria.salvar_nota(ADMIN, "Maria", "nota do admin")
    contas.excluir(PROF)
    con = sqlite3.connect(contas.CONTAS_DB)
    assert con.execute("SELECT COUNT(*) FROM memoria_casos WHERE usuario = ?", (PROF,)).fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM memoria_notas").fetchone()[0] == 1  # só a do admin sobrou
    con.close()
    assert memoria.listar(PROF) == [] and len(memoria.listar(ADMIN)) == 1
