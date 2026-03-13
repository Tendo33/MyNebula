# Pydantic 模型与 Schema 指南

本项目主要使用 Pydantic v2。模型分为两类：

1. **基础模型**：`src/nebula/models/`（通用 BaseModel 与配置）
2. **API Schema**：`src/nebula/schemas/`（请求/响应模型，含 v2 版本化目录）

---

## 目录结构

```text
src/nebula/models/
├── __init__.py
└── base.py              # BaseModel 与通用配置

src/nebula/schemas/
├── __init__.py          # 统一导出常用 Schema
├── graph.py
├── repo.py
├── user.py
└── v2/
    ├── dashboard.py
    ├── data.py
    ├── graph.py
    ├── settings.py
    └── sync.py
```

---

## 基础 BaseModel

`src/nebula/models/base.py` 提供了统一的 Pydantic v2 配置，建议新建模型时优先继承该 BaseModel。

```python
from pydantic import ConfigDict
from nebula.models import BaseModel

class Example(BaseModel):
    name: str

    model_config = ConfigDict(
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )
```

> 说明：现有的 API Schema 有些直接继承 `pydantic.BaseModel`，这是历史遗留写法。新代码建议统一使用 `nebula.models.BaseModel`。

---

## API Schema 约定

### 1. 请求/响应模型位置

- v1：`src/nebula/schemas/*.py`
- v2：`src/nebula/schemas/v2/*.py`

### 2. ORM 响应序列化

Pydantic v2 推荐写法：

```python
from pydantic import ConfigDict
from nebula.models import BaseModel

class RepoResponse(BaseModel):
    id: int
    name: str

    model_config = ConfigDict(from_attributes=True)
```

> 现有代码中可能仍会看到 `class Config: from_attributes = True`（v1 风格），新代码请使用 `ConfigDict`。

---

## 创建新 Schema 示例

```python
from pydantic import Field
from nebula.models import BaseModel

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索关键词", min_length=1)
    limit: int = Field(default=20, ge=1, le=100)

class SearchResponse(BaseModel):
    total: int = Field(..., ge=0)
    items: list[dict] = Field(default_factory=list)
```

---

## 导出与组织

常用 Schema 在 `src/nebula/schemas/__init__.py` 中集中导出，便于外部统一引用：

```python
from nebula.schemas import RepoResponse, GraphData, UserResponse
```

---

## 常见约定与最佳实践

- 使用 `Field(...)` 提供描述与约束。
- 列表/字典等可变默认值使用 `default_factory`。
- Pydantic v2 常用 API：`model_dump()`、`model_validate()`、`model_dump_json()`。
- 避免在 `utils/` 中定义 Schema，统一放在 `schemas/`。

---

## 相关文档

- 环境变量说明：`doc/ENV_VARS.md`
- SDK 使用指南：`doc/SDK_USAGE.md`
