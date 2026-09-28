# 模型

没有单独的模型手册。Embedding 是必填的 OpenAI 兼容接口，LLM 只用于摘要、标签和聚类命名。

看 README 配置概览里的 Embedding 和 LLM 两行，默认值写在 [`.env.example`](../.env.example)：

- `EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`、`EMBEDDING_DIMENSIONS`
- `LLM_BASE_URL`、`LLM_MODEL`、`LLM_OUTPUT_LANGUAGE`

- 中文：[配置概览](../README.zh.md#配置概览)
- English: [Configuration Overview](../README.md#configuration-overview)
