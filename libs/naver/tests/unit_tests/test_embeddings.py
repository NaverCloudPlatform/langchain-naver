"""Test embedding model integration."""

import os

import pytest

from langchain_naver import ClovaXEmbeddings

os.environ["NCP_CLOVASTUDIO_API_KEY"] = "foo"


def test_initialization() -> None:
    """Test embedding model initialization."""
    ClovaXEmbeddings(model="clir-emb-dolphin")


def test_upstage_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ClovaXEmbeddings(
            model="clir-emb-dolphin", model_kwargs={"model": "foo"}
        )


def test_upstage_invalid_model() -> None:
    with pytest.raises(ValueError):
        ClovaXEmbeddings()  # type: ignore[call-arg]


def test_upstage_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ClovaXEmbeddings(model="clir-emb-dolphin", foo="bar")  # type: ignore
    assert llm.model_kwargs == {"foo": "bar"}
