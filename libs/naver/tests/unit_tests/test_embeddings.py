"""Test embedding model integration."""

import os

import pytest

from langchain_naver import ClovaXEmbeddings

os.environ["CLOVASTUDIO_API_KEY"] = "foo"


def test_embeddings_initialization() -> None:
    """Test embedding model initialization."""
    ClovaXEmbeddings()


def test_embeddings_initialization_explict_model() -> None:
    """Test embedding model initialization(explict_model)."""
    ClovaXEmbeddings(model="clir-emb-dolphin")


def test_embeddings_invalid_model_kwargs() -> None:
    with pytest.raises(ValueError):
        ClovaXEmbeddings(model="clir-emb-dolphin", model_kwargs={"model": "foo"})


def test_embeddings_incorrect_field() -> None:
    with pytest.warns(match="not default parameter"):
        llm = ClovaXEmbeddings(model="clir-emb-dolphin", foo="bar")  # type: ignore
    assert llm.model_kwargs == {"foo": "bar"}
