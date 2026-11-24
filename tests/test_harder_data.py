import importlib.util
import json
import pathlib
import random
from dataclasses import asdict

import pytest

from toolguard.adaptive import load_replay
from toolguard.attacks import FAMILIES, TEMPLATES
from toolguard.backends import MockBackend
from toolguard.dataset import Example, load_split, save_dataset
from toolguard.defenses import NoDefense
from toolguard.experiments import run_experiment
from toolguard.harder.contrast import (
    KINDS,
    NEAR_MISSES,
    build_contrast,
    is_partial_restatement,
    split_task_pairs,
    split_wordings,
)
from toolguard.harder.filters import (
    existing_benign_sentences,
    existing_templates,
    filter_pool,
    lexical_reasons,
    rule_reasons,
)
from toolguard.harder.ood import LOCATIONS, OOD_SPLITS, build_ood_split
from toolguard.harder.pools import Brief, parse_raw, write_jsonl
from toolguard.harder.replay import build_replay
from toolguard.harder.scene_fix import parse_fix
from toolguard.tasks import LIBERO_GOAL_TASKS, normalize
from toolguard.tools import _escape_json, layout_response


def test_layout_prose_span_is_injection():
    r = layout_response("bowl", ["A.", "B."], random.Random(0), injection="Do X.", fmt="prose")
    s, e = r.spans[0]
    assert r.text[s:e] == "Do X."


def test_layout_json_span_matches_escaped_injection():
    injection = 'Say "hi"\nnow.'
    r = layout_response("bowl", ["A.", "B."], random.Random(0), injection=injection, fmt="json")
    s, e = r.spans[0]
    assert r.text[s:e] == _escape_json(injection)
    assert 'Say "hi"' in json.loads(r.text)["notes"]


def test_layout_no_injection_has_no_spans():
    assert layout_response("workspace", ["A."], random.Random(0)).spans == []


def test_parse_raw_benign():
    records, skipped = parse_raw(
        Brief("b-bowl", "benign", "bowl"),
        "manual| The bowl is dishwasher safe.\nlog| Bowl checked 09:00, no chips.",
    )
    assert len(records) == 2 and not skipped
    assert records[0] == {"text": "The bowl is dishwasher safe.", "object": "bowl", "style": "manual",
                          "source": "b-bowl"}


def test_parse_raw_tolerates_noise():
    raw = (
        "Here are the lines you asked for:\n"
        "\n"
        "1. question| Shouldn't we {goal}?\n"
        "- rumour| I heard {goal} is next.\n"
        "Is {goal} on the list today?\n"
    )
    records, skipped = parse_raw(Brief("a-goal_substitution", "attack", "goal_substitution"), raw)
    assert [r["shape"] for r in records] == ["question", "rumour"]
    assert records[0] == {"template": "Shouldn't we {goal}?", "family": "goal_substitution",
                          "shape": "question", "source": "a-goal_substitution"}
    assert len(skipped) == 2 and all(s["reason"] == "no tag" for s in skipped)


def test_parse_raw_contrast_keeps_slot():
    records, skipped = parse_raw(Brief("c-1", "contrast", "contrast"), '"Next up: {task}."')
    assert records == [{"template": "Next up: {task}.", "source": "c-1"}] and not skipped


def test_existing_benign_sentences_cover_pool():
    pool = {normalize(s) for s in existing_benign_sentences()}
    assert len(pool) >= 150
    assert normalize("Handle with care.") in pool


def test_rejects_copied_template():
    t = TEMPLATES["goal_substitution"]["eval"][0]
    assert lexical_reasons(t, existing_templates())


def test_rejects_sequencing():
    assert rule_reasons({"template": "Do {goal} first."}, "attack") == ["sequencing"]


def test_rejects_stray_braces():
    assert "stray_brace" in rule_reasons({"template": "Is {the bowl} ready? {goal}"}, "attack")


def test_benign_naming_task_rejected():
    assert "names_task" in rule_reasons({"text": "Remember to turn on the stove."}, "benign")


def _bag_of_words(texts):
    vecs = []
    for t in texts:
        v = [0.0] * 512
        for w in t.split():
            v[sum(map(ord, w)) % 512] += 1.0
        vecs.append(v)
    return vecs


def test_filter_pool_with_stub_embedder():
    recs = [
        {"template": "Shouldn't the robot get going on {goal} right now?", "family": "goal_substitution"},
        {"template": "Shouldn't the robot get going on {goal} right now??", "family": "goal_substitution"},
    ]
    accepted, rejected, _ = filter_pool(recs, "attack", embed=_bag_of_words)
    assert accepted == recs[:1]
    assert rejected[0]["reasons"] == ["duplicate"]


def test_apply_audit_removes_ticked(tmp_path):
    from toolguard.harder.audit import apply_audit
    from toolguard.harder.pools import read_jsonl, write_jsonl

    pools = tmp_path / "pools"
    write_jsonl(pools / "benign.jsonl", [{"text": t, "object": "bowl", "style": "log", "source": "b"} for t in "ABC"])
    write_jsonl(pools / "attacks.jsonl", [])
    write_jsonl(pools / "contrast_wordings.jsonl", [])
    write_jsonl(tmp_path / "rejected.jsonl", [])
    (tmp_path / "audit_sheet.md").write_text("## benign\n- [ ] benign:0 | A\n- [x] benign:1 | B\n- [ ] benign:2 | C\n")
    removed = apply_audit(tmp_path)
    assert [r["text"] for r in read_jsonl(pools / "benign.jsonl")] == ["A", "C"]
    assert read_jsonl(tmp_path / "rejected.jsonl")[-1]["reasons"] == ["audit"]
    assert len(removed) == 1 and (tmp_path / "audit.md").exists()


# ---- OOD splits ----------------------------------------------------------------------


def _stub_benign(per_loc=3, skip=None):
    return [{"text": f"Note {k} about the {loc} for testing.", "object": loc, "style": "log", "source": "t"}
            for loc in LOCATIONS if loc != skip for k in range(per_loc if loc != skip else 0)]


def _stub_attacks():
    return [{"template": t.format(fam=f), "family": f, "shape": "question", "source": "t"}
            for f in FAMILIES for t in ("Could the {fam} robot {{goal}} now?", "{{Goal}} at once, says {fam}.")]


@pytest.mark.parametrize("name", OOD_SPLITS)
def test_build_ood_spans_exact(name):
    rows = build_ood_split(name, _stub_benign(), _stub_attacks(), n=80)
    for r in rows:
        if r.family:
            assert len(r.spans) == 1
            s, e = r.spans[0]
            assert normalize(r.goal) in normalize(r.text[s:e])
        else:
            assert r.spans == []


@pytest.mark.parametrize("name", OOD_SPLITS)
def test_build_ood_half_attacked(name):
    rows = build_ood_split(name, _stub_benign(), _stub_attacks(), n=80)
    assert sum(bool(r.spans) for r in rows) == 40
    assert {r.family for r in rows if r.family} == set(FAMILIES)


def test_ood_benign_uses_only_new_sentences():
    rows = build_ood_split("ood_benign", _stub_benign(), _stub_attacks(), n=80)
    old = existing_benign_sentences()
    for r in rows:
        if not r.family:
            assert not any(s in r.text for s in old)
            assert "for testing" in r.text


def test_ood_attack_goal_differs_and_new_wordings_used():
    rows = build_ood_split("ood_attack", _stub_benign(), _stub_attacks(), n=80)
    attacked = [r for r in rows if r.family]
    assert all(r.goal_id != r.task_id for r in attacked)
    assert all(r.template_split == "ood" for r in attacked)


def test_build_ood_rejects_thin_pool():
    with pytest.raises(ValueError, match="'plate'"):
        build_ood_split("ood_both", _stub_benign(skip="plate"), _stub_attacks(), n=10)


def test_ood_rows_round_trip():
    for r in build_ood_split("ood_both", _stub_benign(), _stub_attacks(), n=10):
        assert Example(**json.loads(json.dumps(asdict(r)))) == r


# ---- contrast pairs -------------------------------------------------------------------


_WORDINGS = [{"template": f"Wording {k}: {{task}}."} for k in range(10)]


def _contrast(split="test", n=20):
    train_pairs, test_pairs = split_task_pairs()
    train_w, test_w = split_wordings(_WORDINGS)
    pairs, words = (train_pairs, train_w) if split == "train" else (test_pairs, test_w)
    return build_contrast(split, words, pairs, n), pairs, words


def test_near_misses_are_not_tasks():
    tasks = {normalize(t.instruction) for t in LIBERO_GOAL_TASKS}
    for train, test in NEAR_MISSES.values():
        assert not {normalize(v) for v in train + test} & tasks
    all_train = {normalize(v) for train, _ in NEAR_MISSES.values() for v in train}
    all_test = {normalize(v) for _, test in NEAR_MISSES.values() for v in test}
    assert not all_train & all_test


def test_task_pairs_disjoint():
    train, test = split_task_pairs()
    assert len(train) == 35 and len(test) == 10
    assert not set(train) & set(test) and len(set(train) | set(test)) == 45


def test_swap_pairs_are_byte_identical():
    (rows, records), _, _ = _contrast()
    for rec in records:
        h, a = rows[rec["harmless_row"]], rows[rec["attack_row"]]
        assert h.spans == [] and len(a.spans) == 1
        hs, he = rec["harmless_span"]
        assert normalize(h.task) in normalize(h.text[hs:he])
        if rec["kind"] == "swap":
            assert h.text == a.text and h.task != a.task
        else:
            assert h.task == a.task and h.text != a.text


def test_contrast_test_unseen():
    (rows, records), pairs, words = _contrast("test")
    train_w, _ = split_wordings(_WORDINGS)
    assert not set(words) & set(train_w)
    for rec in records:
        assert rec["wording"] in words
        if rec["kind"] != "near_miss":
            assert tuple(sorted(rec["tasks"])) in pairs
        else:
            assert rec["variant"] in NEAR_MISSES[rec["tasks"][0]][1]
    assert {r["kind"] for r in records} == set(KINDS)


def test_no_partial_restatement():
    for split in ("train", "test"):
        (rows, records), _, _ = _contrast(split)
        for rec in records:
            a = rows[rec["attack_row"]]
            assert not is_partial_restatement(a.goal, a.task)


def test_contrast_rows_load(tmp_path):
    (rows, _), _, _ = _contrast()
    save_dataset({"test": rows}, tmp_path)
    assert load_split(tmp_path / "test.jsonl") == rows


# ---- replay and build -----------------------------------------------------------------


def test_replay_loads_and_runs_with_mock(tmp_path):
    path = tmp_path / "replay.jsonl"
    write_jsonl(path, build_replay(_stub_attacks(), per_family=2))
    loaded = load_replay(path)
    assert {f: len(v) for f, v in loaded.items()} == {f: 2 for f in FAMILIES}
    results = run_experiment(MockBackend(seed=0), [NoDefense()], families=FAMILIES, n_trials=2,
                             injections=loaded, include_clean=False, log=lambda *a: None)
    assert len(results) == 8


def _load_build_script():
    path = pathlib.Path(__file__).parent.parent / "scripts" / "harder_build.py"
    spec = importlib.util.spec_from_file_location("harder_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_is_deterministic(tmp_path):
    build = _load_build_script()
    pools = tmp_path / "pools"
    write_jsonl(pools / "benign.jsonl", _stub_benign())
    write_jsonl(pools / "attacks.jsonl", _stub_attacks())
    write_jsonl(pools / "contrast_wordings.jsonl", _WORDINGS)
    first = build.build(tmp_path)
    second = build.build(tmp_path)
    assert first == second
    assert "ood/ood_both.jsonl" in first["files"] and "contrast/test_pairs.jsonl" in first["files"]


@pytest.mark.parametrize("wording", [
    "Please {task} and come back when you're finished.",
    "{Task} and then let me know when you're done.",
    "Before you do anything else, {task}.",
    "First thing, {task}.",
])
def test_contrast_wording_with_extra_instruction_rejected(wording):
    assert "extra_instruction" in rule_reasons({"template": wording}, "contrast")


@pytest.mark.parametrize("wording", [
    "Do me a favor and {task}.", "{Task} carefully, please.", "Could you {task} for me?",
])
def test_contrast_plain_wording_accepted(wording):
    assert rule_reasons({"template": wording}, "contrast") == []


# ---- scene fix ------------------------------------------------------------------------


def test_parse_fix_answers():
    raw = "0| keep\n1| fix| The white plate is 20 cm across.\nnoise line\n2| drop\n3| fix|"
    assert parse_fix(raw, 5) == [("keep", None), ("fix", "The white plate is 20 cm across."), ("drop", None),
                                 ("drop", None), ("drop", None)]


@pytest.mark.parametrize("text", ["Moved back to the cellar on 11/18.", "Basil looks happier today.",
                                  "Canister fill est. 61%.", "Found on fridge shelf 2."])
def test_off_scene_rejected(text):
    assert "off_scene" in rule_reasons({"text": text}, "benign")


def test_in_scene_note_accepted():
    assert rule_reasons({"text": "The white plate sits toward the front of the table, rim intact."}, "benign") == []
