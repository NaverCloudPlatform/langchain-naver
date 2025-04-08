from langchain_naver import __all__

EXPECTED_ALL = ["ChatClovaX", "ClovaXEmbeddings"]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
