# Quality Gates

MyNebula 当前把质量检查分成四层：代码风格、测试、构建验证、离线效果评估。

## 1. 基础代码检查

### 后端

```bash
uv run ruff format
uv run ruff check
uv run pytest
```

### 前端

```bash
npm --prefix frontend run lint
npx --prefix frontend tsc --noEmit
npm --prefix frontend run test
npm --prefix frontend run build
```

### E2E

```bash
npm --prefix frontend run test:e2e
```

## 2. CI 当前强制的门槛

GitHub Actions 会跑：

- Backend Ruff lint
- Backend Ruff format check
- Frontend ESLint
- Frontend TypeScript typecheck
- Frontend Vitest
- Frontend production build

也就是说，本地至少建议在提交前把这些命令同步跑一遍。

## 3. 离线效果评估

和推荐/聚类效果强相关的改动，需要执行：

```bash
uv run python scripts/evals/run_all_quality_checks.py
```

脚本位置：

- `scripts/evals/run_all_quality_checks.py`

输入文件：

- `data/eval/related_goldset.jsonl`
- `data/eval/related_predictions.json`
- `data/eval/cluster_goldset.jsonl`

输出文件：

- `data/eval/quality_report.json`

## 4. 当前硬阈值

`run_all_quality_checks.py` 里当前写死的阈值是：

- `p_at_5 >= 0.80`
- `coverage >= 0.95`
- `cluster_purity >= 0.75`

这里的 `coverage` 不是测试覆盖率，而是相关推荐预测对 goldset anchor 的覆盖率。

## 指标含义

### `p_at_5`

相关推荐前 5 个结果中，命中正样本的平均比例。

### `coverage`

在 goldset 的 anchor repo 中，有多少比例拿到了至少一条预测结果。

### `cluster_purity`

聚类结果中，每个簇被主标签占据的纯度。

## 什么时候必须跑离线评估

建议在这些改动后运行：

- 相关推荐打分逻辑
- embedding 文本构造逻辑
- 聚类算法与参数
- snapshot 构造过程里影响边或簇的逻辑
- README 抽取和摘要写回流程

## 什么时候可以不跑

通常以下改动可以不跑离线评估：

- 纯文档修改
- 样式或前端 UI 微调
- 不影响数据处理逻辑的运维脚本修改

## 失败时怎么处理

如果脚本返回非零退出码，说明至少有一个指标跌破阈值。此时建议：

1. 打开 `data/eval/quality_report.json`
2. 看是相关推荐问题还是聚类问题
3. 回到对应 goldset 和预测逻辑排查
4. 不要靠“调低阈值”掩盖回归
