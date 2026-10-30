/** LLM prompts and tag synonyms, ported verbatim from the original pipeline. */

export type Lang = 'zh' | 'en';

export const SUMMARY_SYSTEM: Record<Lang, string> = {
  "zh": "你是一名资深的开源项目分析师，专注于技术项目的分类和文档撰写。\n\n你的职责：\n1. 深入理解项目的技术架构和核心价值\n2. 用专业、简洁且有区分度的中文描述项目\n3. 提取精准的分类标签\n\n输出要求：\n- 必须以纯 JSON 格式输出\n- 不要添加 markdown 代码块标记\n- 不要添加任何额外说明文字",
  "en": "You are a senior technical open-source project analyst focused on software classification and documentation.\n\nYour responsibilities:\n1. Understand project architecture and core value quickly\n2. Write concise, professional, and distinctive English summaries\n3. Extract precise classification tags\n\nOutput requirements:\n- Return pure JSON only\n- Do not wrap output in markdown code fences\n- Do not include any extra commentary"
};

export const SUMMARY_PROMPT: Record<Lang, string> = {
  "zh": "分析以下 GitHub 项目，生成结构化的摘要和标签。\n\n## 项目信息\n{context}\n\n## 输出要求\n\n### summary (摘要)\n- 字数：100-150 字\n- 内容：核心功能、技术特点、适用场景\n- 风格：专业流畅，避免\"这是一个\"等冗余开头\n- 语言：中文\n- 必须包含至少 2 个来自输入信息的具体事实（例如命令、模块、协议、模型、文件名、接口或关键能力）\n- 必须说明项目解决的问题，并指出与同类项目的差异点\n- 禁止空泛套话，避免与其他仓库摘要雷同\n\n### tags (标签)\n- 数量：3-7 个\n- 长度：每个 2-6 字\n- 覆盖：功能类型、技术领域、应用场景\n- 语言：中文优先，技术专有名词可用英文\n\n## 输出格式 (严格按此 JSON 格式)\n{\"summary\": \"项目摘要内容\", \"tags\": [\"标签1\", \"标签2\", \"标签3\"]}",
  "en": "Analyze the GitHub project below and generate a structured summary and tags.\n\n## Repository Info\n{context}\n\n## Output Requirements\n\n### summary\n- length: 70-120 words\n- content: core functionality, technical highlights, and practical use cases\n- style: professional and concise; avoid boilerplate openings\n- language: English\n- Must include at least 2 concrete project details from the input (for example commands, modules, APIs, protocols, models, file paths, or key features)\n- Must state the problem the project solves and one differentiator versus similar tools\n- Avoid generic claims and reusable template phrases\n\n### tags\n- count: 3-7\n- length: 1-4 words each\n- coverage: function category, technical domain, and use case\n- language: English preferred\n\n## Output Format (strict JSON)\n{\"summary\": \"Project summary content\", \"tags\": [\"tag1\", \"tag2\", \"tag3\"]}"
};

export const CLUSTER_SYSTEM: Record<Lang, string> = {
  "zh": "你是一个技术分析专家，擅长分析和分类 GitHub 项目。你的任务是为每个集群生成独特且有区分度的名称。请用中文回答，只输出要求的格式，不要输出其他内容。",
  "en": "You are a technical analyst specializing in GitHub project taxonomy. Return only the requested single-line format."
};

export const CLUSTER_PROMPT: Record<Lang, string> = {
  "zh": "分析以下 GitHub 仓库集合，生成一个简洁且有区分度的集群描述。\n\n仓库列表 ({count} 个):\n{repos}\n\n主要编程语言: {languages}\n常见标签: {topics}{exclusion}\n\n请按以下格式输出（使用 | 分隔，只输出一行）:\n名称|描述|关键词1,关键词2,关键词3\n\n要求:\n1. 名称: 2-5 个中文词，准确概括这组仓库的**核心差异化主题**（不要使用宽泛的\"LLM\"、\"AI\"等词，要具体到应用场景或技术领域）\n2. 描述: 一句话说明这组仓库区别于其他集群的独特共同特点\n3. 关键词: 3-5 个关键词，尽量具体\n\n示例输出格式:\nRAG检索增强|基于向量检索的大模型知识增强框架|RAG,向量检索,知识库,LangChain\n代码生成助手|AI驱动的代码自动生成与补全工具|代码生成,Copilot,IDE插件",
  "en": "Analyze the following GitHub repositories and generate a concise, distinctive cluster description.\n\nRepositories ({count}):\n{repos}\n\nPrimary languages: {languages}\nCommon topics: {topics}{exclusion}\n\nReturn one line only with this format (use | as separator):\nName|Description|keyword1,keyword2,keyword3\n\nRequirements:\n1. Name: 2-5 words, specific and differentiating (avoid broad labels like \"AI\" or \"LLM\")\n2. Description: one sentence describing what makes this cluster unique\n3. Keywords: 3-5 specific keywords\n"
};

export const EXCLUSION: Record<Lang, string> = {
  "zh": "\n\n已有的集群名称（你必须避免使用相同或相似的名称）:\n{names}",
  "en": "\n\nExisting cluster names (you must avoid reusing or paraphrasing them):\n{names}"
};

export const README_HEADINGS = ["api", "feature", "features", "installation", "overview", "quick start", "quickstart", "usage", "功能", "安装", "快速开始", "概览", "用法"];

export const TAG_SYNONYMS: Record<string, string> = {
  "agent-memory": "agent-memory",
  "agent_memory": "agent-memory",
  "long-term-memory": "agent-memory",
  "longterm-memory": "agent-memory",
  "rag-memory": "agent-memory",
  "memory-augmented": "agent-memory",
  "mem0": "agent-memory",
  "记忆": "agent-memory",
  "长期记忆": "agent-memory",
  "retrieval-augmented-generation": "rag",
  "retrieval-augmented": "rag",
  "检索增强": "rag",
  "检索增强生成": "rag",
  "ai-agent": "agent",
  "ai-agents": "agent",
  "agents": "agent",
  "智能体": "agent",
  "agentic": "agent",
  "large-language-model": "llm",
  "大语言模型": "llm",
  "distributed-training-framework": "distributed-training",
  "distributed-systems": "distributed-training",
  "deepspeed": "distributed-training",
  "fsdp": "distributed-training",
  "megatron": "distributed-training"
};
