# Current Frontend

## 什么时候读

- 要改前端代码前先读这里
- 需要确认当前页面、数据面和共享状态边界时读这里

## 当前真实实现

`frontend/` 当前是一个真实产品前端，而不是 starter 页面。

已经落地的主要页面与能力：

- Dashboard：概览、趋势和聚合统计
- Data：仓库列表、过滤和搜索
- Graph：图谱视图、渐进式边加载、时间线和详情侧栏
- Settings：管理员登录、同步、计划任务和运维控制
- Command Palette：共享搜索语义和快捷访问
- i18n：中英文翻译资源

## 当前前端组织

- `frontend/src/api`：HTTP 客户端与 v2 API 适配层
- `frontend/src/components`：图谱、布局和 UI 组件
- `frontend/src/contexts`：认证、图谱上下文与过滤逻辑
- `frontend/src/features`：按业务能力组织的 query hooks
- `frontend/src/pages`：页面级容器
- `frontend/src/utils`：搜索、格式化等共享工具
- `frontend/src/types`：API 与图谱类型

## 当前前端边界

- Dashboard、Data、Graph、Settings 是当前四个关键数据面
- Graph 页面拥有活动快照视图，不应让其他页面重复抓整份图快照
- Graph、Data、Command Palette 的搜索语义必须保持一致
- Settings 轮询必须与组件生命周期和认证状态绑定
- Graph 边数据采用渐进加载，并在自动预取达到阈值后切换为用户显式继续加载
- 图谱过滤已经为边集建立节点到边的索引，在“小可见子集”场景下优先走索引过滤，避免每次都全量扫描边数组
- Settings 中 full refresh 的 `partial_failed` 需要保留 warning 语义，不能在步骤 UI 上伪装成完全成功
- Settings 的同步步骤映射已经拆到 `frontend/src/pages/settings/progress.ts`，页面容器应尽量只保留交互编排

## 当前高价值事实源

- Dashboard 查询：`frontend/src/features/dashboard/hooks/useDashboardQuery.ts`
- Data 查询：`frontend/src/features/data/hooks/useDataReposQuery.ts`
- Graph 查询：`frontend/src/features/graph/hooks/*`
- 图谱过滤：`frontend/src/contexts/graphFiltering.ts`
- 共享搜索：`frontend/src/utils/search.ts`
- Settings 轮询：`frontend/src/pages/settings/polling.ts`

## 共享参考

- 前端规范：[../standards/frontend.md](../standards/frontend.md)
- 视觉与交互基线：[../standards/design-system.md](../standards/design-system.md)
- 验证命令：[../reference/verification.md](../reference/verification.md)
