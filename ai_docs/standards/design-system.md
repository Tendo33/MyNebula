# Design System Standards

## 什么时候读

- 修改前端视觉、组件样式、交互体验或页面层级前先读这里

## 当前视觉基线

- MyNebula 是数据探索产品，不是营销首页
- 视觉重点是信息层级清晰、图谱探索稳定、设置页可操作性强
- 图谱、列表、详情侧栏和命令面板要保持一套统一语言，而不是各自为政

## 默认设计规则

- 先保证信息密度和层级，再谈装饰性
- Graph 相关交互优先服务筛选、定位和理解，不增加噪音动效
- Dashboard 和 Data 的聚合信息要和图谱语义保持一致
- 状态色、聚类色和警告色要有明确职责，不混用

## 交互规则

- 搜索、筛选、时间线和详情联动必须可预测
- 加载态、空态和错误态要明确可见
- 设置页里的高风险操作要清楚暴露上下文和后果

## 文案与语言

- 任何新文案默认同时考虑 `frontend/src/locales/en` 与 `frontend/src/locales/zh`
- 不要在一个页面里混用不同术语描述同一概念，例如 sync job、pipeline run、full refresh

## 共享参考

- 当前前端事实：[../current/frontend.md](../current/frontend.md)
- 前端工程规范：[frontend.md](frontend.md)
