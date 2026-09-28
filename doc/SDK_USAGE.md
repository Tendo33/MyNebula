# SDK

仓库里没有单独发布的客户端 SDK。读数据和触发同步都走 HTTP API。

- 中文：[API 快速参考](../README.zh.md#api-快速参考)
- English: [API Quick Reference](../README.md#api-quick-reference)

本地进程用 `uv run uvicorn nebula.main:app --reload --port 8000` 启动。这是应用入口，不是给外部项目安装的包。
