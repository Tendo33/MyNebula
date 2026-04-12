# Frontend Standards

## 什么时候读

- 开始任何前端改动前先读这里
- 需要确认数据获取、共享搜索和高开销视图边界时回到这里

## 当前默认约束

- 查询数据通过 `frontend/src/api` 和 feature hooks 获取
- Graph 页面拥有活动快照视图，其他页面不要为了轻量元数据重复抓图快照
- Graph、Data、Command Palette 的搜索语义必须统一
- Settings 轮询必须可中止，并与组件生命周期、认证变化绑定

## 不可打破的规则

- 不要在不同页面各写一套搜索匹配逻辑
- 不要在渲染热点组件里内联重计算过滤逻辑
- 不要轻易改 `GraphContext` 的公共形状
- 不要为了页面方便而增加重复 API 请求，先看能否扩充轻量响应

## 组件与页面默认值

- 页面容器负责组合数据和状态
- 共享纯函数放 `utils/`
- 页面之间共享的行为优先下沉到 hooks、context 或工具函数
- i18n 文案改动要同时覆盖中英文资源

## 共享参考

- 当前前端事实：[../current/frontend.md](../current/frontend.md)
- 视觉与交互基线：[design-system.md](design-system.md)
- 验证命令：[../reference/verification.md](../reference/verification.md)
