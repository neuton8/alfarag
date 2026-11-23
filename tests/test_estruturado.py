"""Saída estruturada (JSON por esquema): esquema, texto montado pelo código, fluxo do pipeline e payload do Ollama.
Tudo offline: nenhum teste chama o Ollama."""
import json

import httpx
import pytest

from app import estruturado, harness, llm, pipeline, prompts
from app.estruturado import EstruturadoInvalido

ROTULOS = {"BNCC": "BNCC (2018)", "RAMLI": "Ramli et al. (2021)", "VID": "Experiência pessoal (resumo de vídeo)"}


def _ref(doc, i, texto="A consciência fonológica é habilidade de base para a alfabetização."):
    return {"id": f"{doc}-{i}", "doc": doc, "texto": texto, "tokens": 12, "secao": "", "pagina_ini": i, "pagina_fim": i,
            "cos": 0.9, "rrf": 0.03, "rank_denso": 1, "rank_bm25": 1, "codigos_bncc": []}


def _relato(i, canal="Ludimila Lima"):
    return {**_ref("VID", i, "A professora usa cartelas de sílabas com a turma."), "pagina_ini": "s/p", "pagina_fim": "s/p",
            "secao": f"vídeo v00{i}, {canal}, “Título do vídeo”"}


# janela: [1] BNCC (oficial), [2] RAMLI (científico), [3] e [4] relatos
JANELA = [_ref("BNCC", 3), _ref("RAMLI", 7), _relato(3), _relato(4, "Prof. Nágida")]


def _dados_misto(**sobre):
    d = {
        "orientacao": {"texto": "Trabalhe a consciência fonológica antes das letras, com rimas e sílabas orais.", "refs": [1]},
        "atividade": {"texto": "Use cartelas de sílabas para formar palavras em duplas, em sessões curtas.", "refs": [1, 2]},
        "acompanhamento": {"texto": "Registre a cada semana quais letras o aluno já reconhece sem ajuda.", "refs": [2]},
        "relatos": [
            {"canal": "Ludimila Lima", "texto": "Ludimila Lima relata que usa cartelas de sílabas com a turma.", "refs": [3],
             "fundamentacao": "Sim", "refs_fundamentacao": [1]},
            {"canal": "Prof. Nágida", "texto": "Segundo Prof. Nágida, a turma lê em voz alta em duplas.", "refs": [4],
             "fundamentacao": "Não encontrado nas referências", "refs_fundamentacao": []},
        ],
    }
    d.update(sobre)
    return d


# ---------- esquema ----------
def test_esquema_misto_restringe_quem_pode_citar_quem():
    e = estruturado.esquema("referencia+experiencia", JANELA)
    assert list(e["properties"]) == ["orientacao", "atividade", "acompanhamento", "relatos"] and e["required"] == list(e["properties"])
    for campo in ("orientacao", "atividade", "acompanhamento"):  # seções de referência: só [1] e [2]
        assert e["properties"][campo]["properties"]["refs"]["items"]["enum"] == [1, 2]
    rel = e["properties"]["relatos"]
    assert rel["maxItems"] == 2 and rel["items"]["properties"]["refs"]["items"]["enum"] == [3, 4]  # relato só cita relato
    assert rel["items"]["properties"]["refs_fundamentacao"]["items"]["enum"] == [1, 2]
    assert rel["items"]["properties"]["fundamentacao"]["enum"] == ["Sim", "Em parte", "Não encontrado nas referências"]


def test_esquema_nao_usa_limite_apertado_de_tamanho():
    """maxLength apertado faz o Ollama cortar a frase no meio: o teto é só de segurança."""
    texto = estruturado.esquema("referencia", JANELA[:2])["properties"]["orientacao"]["properties"]["texto"]
    assert texto["maxLength"] >= 1000 and texto["minLength"] <= 30


def test_esquema_referencia_nao_tem_relatos():
    e = estruturado.esquema("referencia", JANELA[:2])
    assert "relatos" not in e["properties"] and e["properties"]["orientacao"]["properties"]["refs"]["items"]["enum"] == [1, 2]


def test_esquema_experiencia_so_relatos_e_sem_fundamentacao():
    e = estruturado.esquema("experiencia", JANELA[2:])
    assert list(e["properties"]) == ["relatos"]
    campos = e["properties"]["relatos"]["items"]["properties"]
    assert set(campos) == {"canal", "texto", "refs"} and campos["refs"]["items"]["enum"] == [1, 2]


def test_modo_misto_sem_relato_na_janela_vira_referencia():
    assert estruturado.modo_efetivo("referencia+experiencia", JANELA[:2]) == "referencia"
    assert "relatos" not in estruturado.esquema("referencia+experiencia", JANELA[:2])["properties"]


def test_modo_unico_trata_toda_fonte_como_referencia():
    assert estruturado.esquema("unico", JANELA)["properties"]["orientacao"]["properties"]["refs"]["items"]["enum"] == [1, 2, 3, 4]


# ---------- parse ----------
@pytest.mark.parametrize("bruto,esperado", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('  {"a": 1}  ', {"a": 1}),
    ('{"orientacao": {"texto": "cortado no mei', None),  # truncado = inválido (o pipeline tenta de novo)
    ("[1, 2]", None),
    ("Resposta em texto livre.", None),
    ("", None),
])
def test_parse(bruto, esperado):
    assert estruturado.parse(bruto) == esperado


# ---------- renderizar ----------
def test_texto_montado_passa_no_harness_sem_nenhum_alerta():
    texto = estruturado.renderizar(_dados_misto(), JANELA, "referencia+experiencia", ROTULOS)
    for titulo in ("**Orientação**", "**Atividade em sala**", "**Acompanhamento**", "**Relatos de experiência**", "**Fontes**"):
        assert titulo in texto
    final, v = harness.verificar_saida(texto, JANELA, False, "stop", "referencia+experiencia", ROTULOS)
    assert v.alertas == [] and v.veredito == "ok"
    assert v.relatos_fora_da_secao == [] and not v.secao_relatos_ausente
    assert v.fundamentacao_corrigida == 0 and v.fundamentacao_tipo_divergente == 0
    assert final == texto  # o harness não precisou reescrever nada


def test_relato_so_aparece_na_secao_de_relatos():
    texto = estruturado.renderizar(_dados_misto(), JANELA, "referencia+experiencia", ROTULOS)
    antes = texto[: texto.index("**Relatos de experiência**")]
    assert "[3]" not in antes and "[4]" not in antes


def test_fontes_listam_so_o_que_foi_citado():
    texto = estruturado.renderizar(_dados_misto(), JANELA, "referencia+experiencia", ROTULOS)
    fontes = texto[texto.index("**Fontes**"):]
    assert "[1] BNCC (2018), p. 3" in fontes and "[2] Ramli et al. (2021), p. 7" in fontes
    assert "[3] Relato de experiência — canal: Ludimila Lima" in fontes and "[4] Relato de experiência — canal: Prof. Nágida" in fontes
    onde_so_ref = estruturado.renderizar({**_dados_misto(), "relatos": _dados_misto()["relatos"][:1]}, JANELA, "referencia+experiencia", ROTULOS)
    assert "[4]" not in onde_so_ref


def test_citacao_solta_no_texto_e_trocada_pelos_refs():
    d = _dados_misto(orientacao={"texto": "Comece pelas rimas [9] e pelas sílabas [1, 2] orais com a turma.", "refs": [2]})
    texto = estruturado.renderizar(d, JANELA, "referencia+experiencia", ROTULOS)
    orientacao = texto.split("**Atividade em sala**")[0]
    assert "[9]" not in orientacao and "[1, 2]" not in orientacao and orientacao.rstrip().endswith("[2]")


def test_fundamentacao_sim_sem_referencia_vira_nao_encontrado():
    d = _dados_misto()
    d["relatos"][0]["refs_fundamentacao"] = []
    texto = estruturado.renderizar(d, JANELA, "referencia+experiencia", ROTULOS)
    assert texto.count("*Fundamentado nas referências: Não encontrado nas referências*") == 2


def test_tipo_da_fundamentacao_vem_do_grupo_da_fonte_nao_do_modelo():
    d = _dados_misto()
    d["relatos"][0].update({"fundamentacao": "Em parte", "refs_fundamentacao": [2, 1]})
    texto = estruturado.renderizar(d, JANELA, "referencia+experiencia", ROTULOS)
    assert "*Fundamentado nas referências: Em parte — produção científica [2], documento oficial [1]*" in texto
    _, v = harness.verificar_saida(texto, JANELA, False, "stop", "referencia+experiencia", ROTULOS)
    assert v.fundamentacao_tipo_divergente == 0


def test_canal_vira_sujeito_so_quando_falta_sujeito():
    f = estruturado._com_canal_como_sujeito
    assert f("Nova Escola", "a turma lê em duplas.") == "Nova Escola relata que a turma lê em duplas."
    assert f("Nova Escola", "A professora usa cartelas.") == "Nova Escola relata que a professora usa cartelas."
    assert f("Nova Escola", "Ludimila Lima relata que usa cartelas.") == "Ludimila Lima relata que usa cartelas."  # já tem sujeito
    assert f("Nova Escola", "Segundo Prof. Nágida, a turma lê.") == "Segundo Prof. Nágida, a turma lê."
    assert f("Nova Escola", "O canal Nova Escola mostra que…") == "O canal Nova Escola mostra que…"


def test_modo_experiencia_inicia_com_o_aviso_e_o_harness_nao_precisa_inserir():
    d = {"relatos": [{"canal": "Prof. Nágida", "texto": "Segundo Prof. Nágida, a turma lê em voz alta em duplas.", "refs": [1]}]}
    texto = estruturado.renderizar(d, [_relato(4, "Prof. Nágida")], "experiencia", ROTULOS)
    assert texto.startswith(estruturado.AVISO_SO_RELATOS) and "Não encontrado nas referências*" in texto
    _, v = harness.verificar_saida(texto, [_relato(4)], False, "stop", "experiencia", ROTULOS)
    assert not v.aviso_experiencia_forcado and v.alertas == []


@pytest.mark.parametrize("mexer", [
    lambda d: d.pop("atividade"),
    lambda d: d.update(atividade={"texto": "curto", "refs": [1]}),
    lambda d: d.update(atividade={"texto": "Use cartelas de sílabas para formar palavras em duplas.", "refs": [3]}),  # relato numa seção de referência
    lambda d: d.update(atividade={"texto": "Use cartelas de sílabas para formar palavras em duplas.", "refs": []}),
    lambda d: d.update(relatos=[]),
    lambda d: d.update(relatos="nenhum"),
    lambda d: d.update(relatos=[{"canal": "X", "texto": "Texto de relato bem comprido o bastante.", "refs": [1]}]),  # referência num relato
])
def test_json_sem_conteudo_integro_e_recusado(mexer):
    d = _dados_misto()
    mexer(d)
    with pytest.raises(EstruturadoInvalido):
        estruturado.renderizar(d, JANELA, "referencia+experiencia", ROTULOS)


# ---------- pipeline com Ollama falso ----------
class _Indice:
    """Referência (BNCC, RAMLI) e relato (VID) cobertos; um trecho de cada grupo."""

    def buscar(self, pergunta, docs, **kw):
        if "VID" in docs:
            return [_relato(3)]
        return [_ref("BNCC", 3), _ref("RAMLI", 7)]


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr(pipeline, "carregar", lambda exp: _Indice())
    monkeypatch.setattr(pipeline, "digest", lambda m: "abc123")
    monkeypatch.setattr(pipeline, "ESTRUTURADO", True)
    return pipeline


FIM = {"prompt_tokens": 2000, "resposta_tokens": 300, "duracao_s": 1, "tok_s": 50, "done_reason": "stop"}


def _json_do_esquema(formato):
    """O que um modelo obediente devolveria: preenche cada campo usando só os números que o esquema permite."""
    def texto_ref(c): return {"texto": "Trabalhe rimas e sílabas orais antes de pedir a escrita das letras.", "refs": c["properties"]["refs"]["items"]["enum"][:1]}
    p = formato["properties"]
    out = {k: texto_ref(p[k]) for k in ("orientacao", "atividade", "acompanhamento") if k in p}
    if "relatos" in p:
        it = p["relatos"]["items"]["properties"]
        r = {"canal": "Ludimila Lima", "texto": "Ludimila Lima relata que usa cartelas de sílabas com a turma.", "refs": it["refs"]["items"]["enum"][:1]}
        if "fundamentacao" in it:
            r.update(fundamentacao="Sim", refs_fundamentacao=it["refs_fundamentacao"]["items"]["enum"][:1])
        out["relatos"] = [r]
    return json.dumps(out, ensure_ascii=False)


class Ollama:
    """chat_stream falso: guarda as chamadas e responde com o que a função `como` mandar (JSON ou texto)."""

    def __init__(self, como):
        self.como, self.chamadas = como, []

    def __call__(self, modelo, mensagens, temperatura=None, chave=None, formato=None, max_tokens=None):
        self.chamadas.append({"temperatura": temperatura, "formato": formato, "max_tokens": max_tokens, "mensagens": mensagens})
        bruto, fim = self.como(len(self.chamadas), formato)
        for i in range(0, len(bruto), 12):  # em pedaços, como o streaming de verdade
            yield {"delta": bruto[i:i + 12]}
        yield {"fim": fim}


def _rodar(pergunta="Como trabalhar a consciência fonológica com o 2º ano?", modelo="qwen3.5:2b-q4_K_M", **kw):
    return list(pipeline.responder(pergunta, modelo=modelo, **kw))


def test_pipeline_json_monta_o_texto_e_registra_no_traco(pipe, monkeypatch):
    ollama = Ollama(lambda n, f: (_json_do_esquema(f), FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    eventos = _rodar(temperatura=0.0)
    final = eventos[-1]
    assert final["desfecho"] == "respondida" and final["verificacao"]["alertas"] == []
    t = final["traco"]
    assert t["saida"] == "json" and t["estruturado"] == {"tentativas": [{"n": 1, "motivo": None, "resposta_tokens": 300}], "reserva_texto": False}
    assert t["versao_prompt"] == "v2.1" and json.loads(t["resposta_bruta"])["orientacao"]["refs"] == [1]
    assert "**Relatos de experiência**" in final["resposta"] and "*Fundamentado nas referências: Sim — documento oficial [1]*" in final["resposta"]
    # o Ollama recebeu o esquema, com T=0 na primeira tentativa
    assert ollama.chamadas[0]["formato"]["required"][0] == "orientacao" and ollama.chamadas[0]["temperatura"] == 0.0
    # a UI recebe o texto montado em UM delta; antes dele só eventos de progresso (que deixam o "Parar" funcionar)
    deltas = [e for e in eventos if e["tipo"] == "delta"]
    assert len(deltas) == 1 and deltas[0]["texto"] == final["resposta"]
    assert any(e["tipo"] == "progresso" for e in eventos)
    # tokens do prompt: estimado (4 caracteres/token) vs real informado pelo Ollama
    assert t["tokens_prompt"]["real"] == 2000 and t["tokens_prompt"]["estimado"] > 0 and t["tokens_prompt"]["razao"] > 0


def test_json_invalido_tenta_de_novo_com_mais_folga(pipe, monkeypatch):
    ollama = Ollama(lambda n, f: ("isto não é JSON" if n == 1 else _json_do_esquema(f), FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    final = _rodar(temperatura=0.0)[-1]
    assert final["traco"]["saida"] == "json" and [x["motivo"] for x in final["traco"]["estruturado"]["tentativas"]] == ["json_invalido", None]
    a, b = ollama.chamadas
    assert b["temperatura"] >= 0.3 > a["temperatura"] == 0.0  # com seed fixa, T=0 repetiria a mesma saída
    assert b["max_tokens"] == 2 * a["max_tokens"]


def test_truncado_conta_como_invalido(pipe, monkeypatch):
    ollama = Ollama(lambda n, f: (_json_do_esquema(f), {**FIM, "done_reason": "length"} if n == 1 else FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    t = _rodar()[-1]["traco"]
    assert [x["motivo"] for x in t["estruturado"]["tentativas"]] == ["truncada", None]


def test_json_sem_fonte_valida_tambem_tenta_de_novo(pipe, monkeypatch):
    sem_fonte = json.dumps({"orientacao": {"texto": "Trabalhe rimas e sílabas orais com a turma.", "refs": []},
                            "atividade": {"texto": "Use cartelas de sílabas em duplas.", "refs": [1]},
                            "acompanhamento": {"texto": "Registre o progresso toda semana.", "refs": [1]}})
    ollama = Ollama(lambda n, f: (sem_fonte if n == 1 else _json_do_esquema(f), FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    t = _rodar()[-1]["traco"]
    assert t["estruturado"]["tentativas"][0]["motivo"].startswith("incompleto") and t["saida"] == "json"


def test_duas_falhas_caem_no_texto_livre_da_v2_0(pipe, monkeypatch):
    texto = "**Orientação**\nTrabalhe rimas [1].\n\n**Fontes**\n[1] BNCC (2018), p. 3"
    ollama = Ollama(lambda n, f: (texto if f is None else "lixo", FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    final = _rodar()[-1]
    t = final["traco"]
    assert final["desfecho"] == "respondida" and t["saida"] == "texto" and t["estruturado"]["reserva_texto"] is True
    assert [c["formato"] is not None for c in ollama.chamadas] == [True, True, False]  # 2 tentativas em JSON, depois texto livre
    assert "Trabalhe rimas" in final["resposta"] and t["resposta_bruta"] == texto
    assert ollama.chamadas[2]["mensagens"][0]["content"] == prompts.PROMPT_SISTEMA  # na reserva volta o prompt v2.0 em texto
    assert ollama.chamadas[0]["mensagens"][0]["content"] == prompts.PROMPT_SISTEMA_JSON


def test_estruturado_desligado_segue_o_fluxo_antigo(pipe, monkeypatch):
    monkeypatch.setattr(pipeline, "ESTRUTURADO", False)
    ollama = Ollama(lambda n, f: ("**Orientação**\nRimas [1].", FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    t = _rodar()[-1]["traco"]
    assert len(ollama.chamadas) == 1 and ollama.chamadas[0]["formato"] is None
    assert t["saida"] == "texto" and "estruturado" not in t


def test_modelo_remoto_nunca_recebe_esquema(pipe, monkeypatch):
    ollama = Ollama(lambda n, f: ("**Orientação**\nRimas [1].", FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    t = _rodar(modelo="deepseek:deepseek-flash")[-1]["traco"]
    assert len(ollama.chamadas) == 1 and ollama.chamadas[0]["formato"] is None and t["saida"] == "texto"


def test_erro_do_modelo_no_modo_json_vira_desfecho_erro(pipe, monkeypatch):
    def quebra(*a, **k):
        raise httpx.ConnectError("Ollama fora do ar")
        yield  # noqa: F841 (torna a função um gerador)
    monkeypatch.setattr(pipeline, "chat_stream", quebra)
    final = _rodar()[-1]
    assert final["desfecho"] == "erro" and "O modelo local não respondeu" in final["resposta"]


def test_fechar_o_gerador_no_meio_da_geracao_para_o_modelo(pipe, monkeypatch):
    """Botão "Parar": a geração em JSON não tem delta para a UI, mas os eventos de progresso deixam o gerador ser fechado."""
    fechado = {"sim": False}

    def lento(*a, **k):
        try:
            for i in range(200):
                yield {"delta": "x" * 10}
        finally:
            fechado["sim"] = True

    monkeypatch.setattr(pipeline, "chat_stream", lento)
    g = pipeline.responder("Como trabalhar a consciência fonológica com o 2º ano?", modelo="qwen3.5:2b-q4_K_M")
    for ev in g:
        if ev["tipo"] == "progresso":
            break
    g.close()
    assert fechado["sim"] is True


def test_pergunta_fora_do_escopo_recusa_sem_chamar_o_modelo(pipe, monkeypatch):
    ollama = Ollama(lambda n, f: ("não deveria ser chamado", FIM))
    monkeypatch.setattr(pipeline, "chat_stream", ollama)
    eventos = _rodar("Como ensinar frações equivalentes para uma turma do 5º ano?")
    final = eventos[-1]
    assert final["desfecho"] == "sem_cobertura" and final["resposta"] == harness.FRASE_SEM_COBERTURA and ollama.chamadas == []
    assert final["traco"]["decisao"]["fora_escopo"].startswith("série fora do escopo")
    assert any(e["tipo"] == "contexto" for e in eventos)  # a UI mostra a janela e a recusa como em qualquer falta de cobertura


# ---------- payload do Ollama ----------
class _Resp:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def raise_for_status(self): pass
    def iter_lines(self):
        yield json.dumps({"message": {"content": "{}"}, "done": False})
        yield json.dumps({"message": {"content": ""}, "done": True, "eval_count": 1, "eval_duration": 1, "total_duration": 1,
                          "done_reason": "stop"})


def test_payload_do_stream_leva_format_e_num_predict(monkeypatch):
    vistos = []
    monkeypatch.setattr(httpx, "stream", lambda metodo, url, **kw: vistos.append(kw["json"]) or _Resp())
    esq = {"type": "object", "properties": {"a": {"type": "string"}}}
    list(llm.chat_stream("qwen3.5:2b-q4_K_M", [{"role": "user", "content": "oi"}], 0.0, formato=esq, max_tokens=777))
    list(llm.chat_stream("qwen3.5:2b-q4_K_M", [{"role": "user", "content": "oi"}], 0.0))
    com, sem = vistos
    assert com["format"] == esq and com["options"]["num_predict"] == 777 and com["think"] is False and com["stream"] is True
    assert "format" not in sem and sem["options"]["num_predict"] == llm.MAX_RESPOSTA  # fluxo antigo intacto
