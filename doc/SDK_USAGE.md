# 程序化使用指南

MyNebula 的主入口仍然是 HTTP API 和 Web UI，但当前仓库也支持以 Python 包的方式被脚本或其他 ASGI 应用调用。

需要先说明两点：

1. 这个项目目前不是一个“稳定对外发布的通用 SDK”
2. 比较稳定的程序化入口主要是应用工厂、配置读取和少数脚本工具

如果你的目标是读取业务数据或驱动前端页面，优先仍然应该走 HTTP API，尤其是 `/api/v2/dashboard`、`/api/v2/data/repos`、`/api/v2/graph*` 和 `/api/v2/settings*` 这些已经稳定下来的聚合接口。

## 安装方式

本地开发推荐 editable install：

```bash
uv sync --all-extras
uv pip install -e .
```

安装后导入名是 `nebula`，不是 `mynebula`。

## 最常用的程序化入口

### 1. 创建 ASGI 应用

```python
from nebula.main import create_app

app = create_app()
```

适合：

- 挂到自定义 ASGI 进程
- 在集成测试里创建应用实例
- 检查路由是否已注册

### 2. 读取配置

```python
from nebula.core.config import (
    get_app_settings,
    get_database_settings,
    get_embedding_settings,
)

app_settings = get_app_settings()
db_settings = get_database_settings()
embedding_settings = get_embedding_settings()
```

这些 getter 都带 `lru_cache`，适合在进程内重复读取。

如果你在运行时改了环境变量并希望重新加载，可以调用：

```python
from nebula.core.config import reload_all_settings

reload_all_settings()
```

### 3. 运行离线质量检查

```python
from scripts.evals.run_all_quality_checks import run_checks

metrics = run_checks()
print(metrics)
```

### 4. 调用工具函数

项目仍保留一批通用工具函数：

```python
from nebula.utils import get_logger

logger = get_logger(__name__)
logger.info("hello from MyNebula")
```

## 一个最小示例

```python
from nebula.main import create_app
from nebula.core.config import get_app_settings

app = create_app()
settings = get_app_settings()

print(app.title)
print(settings.app_version)
```

## 为什么导入时不用写 `src.nebula`

仓库采用标准 `src` 布局：

```text
MyNebula/
├── src/
│   └── nebula/
├── tests/
└── pyproject.toml
```

这意味着：

- 安装包后，直接 `import nebula`
- 测试里也直接 `import nebula`
- 不要写 `from src.nebula ...`

测试之所以也能这样写，是因为 `pyproject.toml` 里给 pytest 配了：

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
```

## 路径和导入路径不要混淆

### 导入路径

`pythonpath = ["src"]` 只影响 Python 找模块的位置。

### 文件路径

日志、数据文件、临时文件仍然和当前工作目录有关，而不是和 `src/` 有关。

例如：

- 在项目根目录执行命令，`logs/app.log` 会相对项目根目录解析
- 换一个目录执行脚本，相对路径结果就会变

所以在脚本里建议优先使用 `pathlib.Path` 构造绝对路径。

## 当前不建议当作稳定 SDK 的部分

下面这些更适合作为项目内部模块使用，而不是对外承诺稳定接口：

- `application/services/*`
- `core/*` 中的具体 provider 实现
- `db/models.py` 里直接操作持久化细节
- 前端 `api/v2/*.ts` 之外的实现细节

如果你只是想集成 MyNebula，优先使用：

- Web UI
- `/api/v2` HTTP API
- 现成脚本命令

其中比较值得直接依赖的 v2 契约有：

- `/api/v2/dashboard`
  - 直接返回 summary、top languages、top topics、top clusters
- `/api/v2/data/repos`
  - 直接返回分页仓库列表、轻量 cluster 元数据、`total_repos`
- `/api/v2/graph`
  - 返回当前 active snapshot 的图数据
- `/api/v2/graph/edges`
  - 用于大图分页边加载

## 相关文档

- `README.md`
- `doc/ENV_VARS.md`
- `doc/MODELS_GUIDE.md`
