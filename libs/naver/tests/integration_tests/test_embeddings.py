"""Test Naver embeddings."""

from langchain_naver.embeddings import ClovaXEmbeddings


def test_embedding_documents() -> None:
    """Test ClovaX embeddings."""
    documents = [
        "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 수 있는 "
        "개발 도구입니다.",
        "LangChain은 언어 모델 기반 애플리케이션 개발을 지원하는 오픈소스 입니다.",
    ]
    embedding = ClovaXEmbeddings()
    output = embedding.embed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0
    assert len(output[1]) > 0


async def test_aembedding_documents() -> None:
    """Test ClovaX embeddings."""
    documents = [
        "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 수 있는 "
        "개발 도구입니다.",
        "LangChain은 언어 모델 기반 애플리케이션 개발을 지원하는 오픈소스 입니다.",
    ]
    embedding = ClovaXEmbeddings()
    output = await embedding.aembed_documents(documents)
    assert len(output) == 2
    assert len(output[0]) > 0
    assert len(output[1]) > 0


def test_embedding_query() -> None:
    """Test ClovaX embeddings."""
    document = (
        "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 "
        "수 있는 개발 도구입니다."
    )
    embedding = ClovaXEmbeddings(model="bge-m3")
    output = embedding.embed_query(document)
    assert len(output) > 0


async def test_aembedding_query() -> None:
    """Test ClovaX embeddings."""
    document = (
        "CLOVA Studio는 HyperCLOVA X 모델을 활용하여 AI 서비스를 손쉽게 만들 "
        "수 있는 개발 도구입니다."
    )
    embedding = ClovaXEmbeddings(model="bge-m3")
    output = await embedding.aembed_query(document)
    assert len(output) > 0
