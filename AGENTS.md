# MyNebula

把 GitHub Stars 变成语义知识星云，发布为静态站点。单包，pnpm，Node 24。

## 形态

- `pipeline/`：离线管线，用 `node` 直接跑类型剥离，不编译、无运行时依赖。`pnpm sync` 拉取 stars，为新仓库或有变化的仓库生成摘要/标签与向量，连接相关仓库，按 Louvain 划分社区（id 与上一轮对齐），写出 `frontend/public/data/graph.json`。
- `data/cache.json.gz` 保存状态，**重跑必须增量**：只有新仓库和内容变化的仓库才能打付费 API。评分输入没变时复用边与分组，但 `graph.json` 仍会重写。
- 只有 `GITHUB_TOKEN` 是必需的。没有 embedding key 时按 topics、star 列表和语言连边；没有 LLM key 时不调用对话模型。拿不到向量的仓库留在图里，不剔除。
- `frontend/`：Vite + 原生 TypeScript + d3-force。Graph（首页）、Data、Settings 与 Cmd/Ctrl+K 面板，hash 路由以便任意静态托管。画布在 `frontend/src/graph/canvas.ts`：边上的力模拟跑至稳定后转为绘制的漂移（隐藏标签页暂停，减弱动效时关闭），拖拽会把漂移写回布局并重新加热。配色与可见性是 `graph/style.ts` 里的纯函数。
- `site.config.json`（仓库根）：fork 可改的站点名称、图标、描述、主题色、默认语言与公开 URL。`vite.config.ts` 在构建时把它注入 `index.html` 与 `virtual:site-config`；`SITE_*` 环境变量优先。运行时通过 `frontend/src/site.ts` 读取，不要在组件里硬编码产品名。
- `.github/workflows/pages.yml` 每天跑一次 sync 并部署到 Pages。

## 规则

- 管线代码必须是可擦除的 TypeScript（不用 enum、namespace、参数属性），Node 只剥离类型不编译。
- 保持前端轻量：不引入 UI 框架或组件库。
- 模板一律经过 `frontend/src/html.ts` 的 `html`，它会对每个插值做转义。条件标记写成 `cond ? html\`…\` : ''`，不要用 `cond && …`，否则布尔值会渲染成 "true"/"false"。
- Graph、Data 页与面板搜索共用 `frontend/src/search.ts`。
- `graph.json` 的字段契约要在 `pipeline/graph.ts` 与 `frontend/src/types.ts` 之间保持同步。
- 前端保留 Graph、Data、Settings 三条流程、Graph 与 Data 的 URL 状态、命令面板、en/zh 双语。
- 视觉沿用 `DESIGN.md` 里的 Vercel Geist：细线优先于卡片，控件 6px（chip、badge、按钮同一套圆角，不用胶囊），一个墨色加若干社区色。
- CSS 里 `html` 的 `font-size` 必须是 16px（rem 的基准）；正文 14px 由 `body` 设置。不要在 `html` 上写 14px，否则全部 rem 尺寸和控件高度会整体缩掉约 14%。

## 命令

```bash
pnpm sync    # 拉取 stars 并重建 graph.json（增量）
pnpm lint    # tsc 检查 pipeline 与 frontend
pnpm test    # vitest
pnpm build
pnpm dev
```
