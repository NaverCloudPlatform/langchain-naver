# langchain-naver

This package contains the LangChain integrations for [Naver Cloud CLOVA Studio](https://clovastudio.ncloud.com/) through their [APIs](https://api.ncloud-docs.com/docs/clovastudio-chatcompletions).

## Installation and Setup

- Install the LangChain partner package
```bash
pip install -U langchain-naver
```

- Get an Naver Cloud CLOVA Studio api key from [issuing it](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary#API%ED%82%A4) and set it as an environment variable (`CLOVASTUDIO_API_KEY`)

## Chat Models

This package contains the `ChatClovaX` class, which is the recommended way to interface with clova studio models.

See a [usage example](https://python.langchain.com/docs/integrations/chat/naver/)

## Embeddings

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/naver)

Use `clir-emb-dolphin` model for embeddings.
