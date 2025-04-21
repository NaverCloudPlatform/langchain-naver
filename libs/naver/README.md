# langchain-naver

This package contains the LangChain integrations for `Naver Cloud` [CLOVA Studio](https://clovastudio.ncloud.com/) through their [APIs](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary).

Please refer to [NCP User Guide](https://guide.ncloud-docs.com/docs/clovastudio-overview) for more detailed instructions (also in Korean).

## Installation and Setup

- Install the dedicated LangChain integration package for Naver

```bash
pip install -U langchain-naver
```

- Get a CLOVA Studio API Key by [issuing it](https://api.ncloud-docs.com/docs/ai-naver-clovastudio-summary#API%ED%82%A4) and set it as an environment variable (`CLOVASTUDIO_API_KEY`).


> (Note) `langchain-community`, a collection of [third-party integrations](https://python.langchain.com/docs/concepts/architecture/#langchain-community) including Naver, is outdated.
> - **Use `langchain-naver` instead as new features should only be implemented via this package**.
> - If you are using `langchain-community` (outdated) and got a legacy API Key (that doesn't start with `nv-*` prefix), you might need to get an additional API Gateway API Key by [creating your app](https://guide.ncloud-docs.com/docs/en/clovastudio-playground01#create-test-app) and set it as `NCP_APIGW_API_KEY`.

## Chat models

This package contains the `ChatClovaX` class, which is the recommended interface to chat models in CLOVA Studio.

### ChatClovaX 

See a [usage example](https://python.langchain.com/docs/integrations/chat/naver/).

## Embedding models

This package contains the `ClovaXEmbeddings` class, which is the recommended interface to embedding models in CLOVA Studio.

### ClovaXEmbeddings

See a [usage example](https://python.langchain.com/docs/integrations/text_embedding/naver).