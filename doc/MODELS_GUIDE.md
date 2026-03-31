# 模型与 Schema 指南

MyNebula 目前有三层“数据模型”需要区分：

1. SQLAlchemy ORM 模型：数据库真实表结构
2. Pydantic API Schema：HTTP 请求和响应
3. 少量通用 BaseModel 封装：提供统一序列化配置

## 目录总览

```text
src/nebula/db/models.py          # SQLAlchemy ORM 模型
src/nebula/models/base.py        # 通用 Pydantic BaseModel
src/nebula/schemas/              # 通用 API schema
src/nebula/schemas/v2/           # v2 路由专用 schema
```

## ORM 模型地图

所有数据库模型都定义在 `src/nebula/db/models.py`。

### 核心实体

| 模型 | 作用 |
| --- | --- |
| `User` | 单用户模式下的用户主体、同步统计、图谱默认配置 |
| `StarredRepo` | 星标仓库主表，包含元数据、README、embedding、聚类结果 |
| `StarList` | GitHub Star List 映射 |
| `Cluster` | 聚类结果、聚类中心、颜色和关键词 |

### 运行与运维实体

| 模型 | 作用 |
| --- | --- |
| `SyncSchedule` | 定时同步配置 |
| `PipelineRun` | 新版同步 pipeline 的运行记录 |
| `SyncTask` | 全量刷新等后台任务状态 |
| `RepoRelatedFeedback` | 相关推荐反馈 |
| `RepoRelatedCache` | 相关推荐缓存 |

### 快照实体

| 模型 | 作用 |
| --- | --- |
| `GraphSnapshot` | 版本化图谱快照主表 |
| `GraphSnapshotNode` | 某次快照的节点 payload |
| `GraphSnapshotEdge` | 某次快照的边 payload |
| `GraphSnapshotTimeline` | 某次快照的时间线 payload |

## Pydantic Schema 组织方式

### `src/nebula/schemas/`

这里放“跨多个场景可复用”的基础响应结构，例如：

- `graph.py`
  - `GraphData`
  - `GraphNode`
  - `GraphEdge`
  - `TimelineData`
- `repo.py`
  - `RepoResponse`
  - `RepoSearchRequest`
  - `RelatedRepoResponse`

### `src/nebula/schemas/v2/`

这里放 `/api/v2` 直接使用的聚合响应：

- `dashboard.py`
- `data.py`
- `graph.py`
- `settings.py`
- `sync.py`

典型例子：

- `SettingsResponse`：设置页一次性加载的数据
- `PipelineStatusResponse`：同步 pipeline 运行状态
- `DataReposResponse`：Data 页面列表查询
- `GraphEdgesPage`：图谱边分页结果

## 什么时候加 ORM，什么时候只加 Schema

### 只加 Schema

适用于：

- 只是新增一个 API 聚合响应
- 只是新增一个前端查询参数
- 数据已经存在于现有表中，不需要持久化新字段

### 必须改 ORM 并新增 migration

适用于：

- 需要持久化新的业务字段
- 要新增任务状态、缓存、快照内容
- 需要改变约束、索引或关系

这类改动必须同步：

- `src/nebula/db/models.py`
- `alembic/versions/*.py`
- 相关 schema
- 相关测试

## 推荐写法

### ORM -> Pydantic 响应

当响应直接来源于 ORM 实体时，推荐显式开启 `from_attributes`：

```python
from pydantic import BaseModel, ConfigDict


class RepoResponse(BaseModel):
    id: int
    full_name: str

    model_config = ConfigDict(from_attributes=True)
```

### 约束和默认值

优先把接口约束写在字段定义上：

```python
from pydantic import BaseModel, Field


class GraphDefaultsUpdateRequest(BaseModel):
    max_clusters: int = Field(..., ge=2, le=30)
    min_clusters: int = Field(..., ge=2, le=30)
```

### 可变默认值

列表、字典统一使用 `default_factory`：

```python
from pydantic import BaseModel, Field


class Example(BaseModel):
    tags: list[str] = Field(default_factory=list)
```

## 当前项目里的几个重要边界

### 图谱读取

- 原始图谱返回结构位于 `src/nebula/schemas/graph.py`
- `/api/v2/graph/edges` 的分页结构位于 `src/nebula/schemas/v2/graph.py`

### 设置页

- 设置聚合响应与 schedule/full-refresh 结构位于 `src/nebula/schemas/v2/settings.py`
- 管理员认证响应定义在 `src/nebula/api/v2/auth.py` 内部

### 同步流程

- 新版同步入口只走 `/api/v2/sync/*`
- pipeline 状态 schema 位于 `src/nebula/schemas/v2/sync.py`

## 新增模型时的建议检查表

- 数据到底要不要落库
- 如果落库，是否需要 migration
- 是否已有现成 schema 可以复用
- 字段命名是否和前端/数据库保持一致
- 是否需要 `from_attributes=True`
- 是否补齐了约束、默认值和测试

## 相关文档

- `doc/ENV_VARS.md`
- `doc/QUALITY_GATES.md`
- `doc/SDK_USAGE.md`
