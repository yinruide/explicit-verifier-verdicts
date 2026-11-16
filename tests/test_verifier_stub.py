"""stub verifier conforms to score_spec.md."""

from __future__ import annotations

import pytest

from rafr.core.interfaces import Verifier
from rafr.verifiers import stub


def test_stub_conforms_to_score_spec():
    v1, v2 = stub.stub_pair()
    assert isinstance(v1, Verifier) and stub.is_stub(v1.verifier_id)
    texts = ["you idiot", "have a nice day", "  you   IDIOT "]
    s = v1.score(texts)
    assert len(s) == 3 and all(0.0 <= x <= 1.0 for x in s)
    assert v1.score(texts) == s  # deterministic
    assert s[0] == s[2]  # whitespace/case normalized
    assert s[0] > s[1]  # lexicon bump
    assert v1.score([]) == []
    assert v1.score(texts) != v2.score(texts)  # heterogeneous pair disagrees somewhere
    with pytest.raises(ValueError):
        v1.score(["x"] * (v1.batch_limit + 1))
    with pytest.raises(ValueError):
        stub.StubVerifier("real-v1")


def test_the_always_toxic_pair_scores_every_text_one_and_is_a_stub():
    v1, v2 = stub.always_toxic_pair()
    for v in (v1, v2):
        assert stub.is_stub(v.verifier_id) and v.describe()["smoke_only"]
        assert v.score(["a", "", "x" * 999]) == [1.0, 1.0, 1.0]
        assert isinstance(v, Verifier)
    assert v1.verifier_id != v2.verifier_id
    with pytest.raises(ValueError):
        stub.ConstantStubVerifier("always-toxic", 1.0)  # not a stub- id
    with pytest.raises(ValueError):
        stub.ConstantStubVerifier("stub-x", 1.5)
