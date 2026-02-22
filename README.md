# AstrBot Enhance Mode - 群聊增强插件

**版本**: v0.1.0 | **作者**: 阿汐

---

## 功能概述

本插件为 AstrBot 的群聊场景提供增强功能，完全替代内置的「群聊上下文感知」和「主动回复」，并额外支持角色标签和发送者 ID。

- **角色显示**: 在系统提示词中注入用户角色（admin/member），让 LLM 感知发送者权限
- **React 模式**: 将请求改写为“群聊反应模式”（基于群聊历史对新消息做反应）
- **增强群聊上下文**: 以包含发送者 ID、角色标签的格式记录群聊消息，并注入 LLM 上下文（依赖 React 模式）
- **图片转述**: 使用 LLM 为群聊中的图片生成文字描述，让纯文本模型也能「看到」图片
- **主动回复**: 按概率随机回复群聊消息（无需被 @），支持白名单限制（依赖 React 模式）

---

## 快速开始

### 安装

将插件文件夹放置于 AstrBot 的 `data/plugins/` 目录下，重启 AstrBot 即可。

### 使用前配置

使用本插件前，请先在 AstrBot 后台**关闭以下内置功能**（避免重复）：

1. **群聊上下文感知**（`group_icl_enable`）→ 关闭
2. **主动回复**（`active_reply.enable`）→ 关闭

以下内置功能**保持开启**：

1. **用户识别**（`identifier`）→ 保持开启（角色显示和发送者 ID 依赖此功能）

然后在插件配置页面启用本插件的对应功能即可。

> **注意**: 如果启用了会话白名单，需要将目标群加入白名单，否则非管理员的群消息会被 pipeline 拦截，无法被本插件记录。

---

## 配置说明

所有配置均可在 AstrBot 控制台的插件配置页面修改。

### 角色显示

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `role_display` | bool | `true` | 在系统提示词的 `<system_reminder>` 中追加 `Role: admin/member` |

启用后，system_reminder 的效果：

```
<system_reminder>User ID: 123456, Nickname: 张三, Role: admin
Group name: 技术交流群
Current datetime: 2026-02-22 21:00 (CST)</system_reminder>
```

### React 模式

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `react_mode.enable` | bool | `false` | 启用后，增强上下文与主动回复能力生效；群聊中的请求会改写为“群聊反应模式” |

启用后，插件在群聊请求中会使用以下模式：

1. 注入群聊历史
2. 将当前消息作为“new message”让模型做即时反应
3. 清空 `req.contexts`，避免与 react 提示词冲突

### 增强群聊上下文

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `group_context.enable` | bool | `false` | 启用增强群聊上下文记录（需先开启 `react_mode.enable`） |
| `group_context.max_messages` | int | `300` | 每个会话保留的最大消息条数 |
| `group_context.include_sender_id` | bool | `true` | 消息格式中包含发送者 ID |
| `group_context.include_role_tag` | bool | `true` | 消息格式中包含 admin/member 角色标签 |
| `group_context.image_caption` | bool | `false` | 使用 LLM 为图片生成文字描述 |
| `group_context.image_caption_provider_id` | string | `""` | 图片转述使用的 LLM 提供商 ID，留空使用默认 |
| `group_context.image_caption_prompt` | string | `"用一句话描述这张图片。"` | 图片转述提示词 |

启用后，群聊消息的记录格式：

```
[张三/123456/21:00:00](admin):  今天天气不错
---
[李四/654321/21:00:05](member):  确实
---
[张三/123456/21:00:10](admin):  [Image]
---
[You/21:00:15]: 是的，今天阳光明媚！
```

### 主动回复

| 配置项 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `active_reply.enable` | bool | `false` | 启用群聊主动回复（需先开启 `react_mode.enable`） |
| `active_reply.possibility` | float | `0.1` | 每条消息的回复概率（0.0 - 1.0） |
| `active_reply.whitelist` | string | `""` | 限制主动回复的群列表，逗号分隔，留空则所有群生效 |

---

## 插件结构

```
astrbot_plugin_astrbot_enhance_mode/
├── main.py              # 插件主逻辑
├── metadata.yaml        # 插件元信息
├── _conf_schema.json    # 配置 Schema（WebUI 自动渲染）
└── README.md            # 说明文档
```
