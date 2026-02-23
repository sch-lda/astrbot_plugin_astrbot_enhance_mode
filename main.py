import asyncio
import datetime
import random
import traceback
import uuid

from astrbot.api import logger, sp, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain, Reply
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart

from .plugin_config import PluginConfig, parse_plugin_config
from .runtime_state import RuntimeState
from .tag_utils import (
    bounded_chat_history_text,
    build_interaction_instructions,
    chain_has_refuse_tag,
    clean_response_text_for_history,
    has_refuse_tag,
    transform_result_chain,
)


class Main(star.Star):
    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.runtime = RuntimeState()

    def _cfg(self) -> PluginConfig:
        return parse_plugin_config(self.config)

    def _touch_origin(self, origin: str, cfg: PluginConfig) -> None:
        self.runtime.touch_origin(origin, cfg.global_settings.lru_cache.max_origins)

    def _allow_active_reply(self, event: AstrMessageEvent, cfg: PluginConfig) -> bool:
        ar = cfg.active_reply
        if not cfg.active_reply_enabled:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if ar.whitelist and (
            event.unified_msg_origin not in ar.whitelist
            and (event.get_group_id() and event.get_group_id() not in ar.whitelist)
        ):
            return False
        return True

    async def _resolve_persona_mask(self, event: AstrMessageEvent) -> tuple[str, str]:
        persona_id = ""
        try:
            session_service_config = await sp.get_async(
                scope="umo",
                scope_id=event.unified_msg_origin,
                key="session_service_config",
                default={},
            )
            if isinstance(session_service_config, dict):
                persona_id = str(session_service_config.get("persona_id") or "").strip()
        except Exception as e:
            logger.debug(f"enhance-mode | 获取 session persona 失败: {e}")

        if not persona_id:
            try:
                curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if curr_cid:
                    conv = await self.context.conversation_manager.get_conversation(
                        event.unified_msg_origin,
                        curr_cid,
                    )
                    if conv and conv.persona_id:
                        persona_id = str(conv.persona_id).strip()
            except Exception as e:
                logger.debug(f"enhance-mode | 获取 conversation persona 失败: {e}")

        if not persona_id:
            cfg = self.context.get_config(umo=event.unified_msg_origin)
            persona_id = str(
                cfg.get("provider_settings", {}).get("default_personality") or ""
            ).strip()

        if persona_id == "[%None]":
            return "none", "No persona mask."

        persona = None
        if persona_id:
            try:
                persona = next(
                    (
                        p
                        for p in self.context.persona_manager.personas_v3
                        if p.get("name") == persona_id
                    ),
                    None,
                )
            except Exception:
                persona = None

        if not persona:
            try:
                persona = await self.context.persona_manager.get_default_persona_v3(
                    event.unified_msg_origin
                )
            except Exception:
                persona = {"name": "default", "prompt": ""}

        persona_name = str(persona.get("name") or "default")
        persona_prompt = str(persona.get("prompt") or "").strip()
        if not persona_prompt:
            persona_prompt = "You are a helpful and friendly assistant."
        return persona_name, persona_prompt

    async def _judge_model_choice(
        self,
        event: AstrMessageEvent,
        cfg: PluginConfig,
        origin: str,
        messages: list[str],
        trigger_reason: str,
    ) -> bool:
        ar = cfg.active_reply
        history = self.runtime.model_choice_histories[origin]
        history_context_lines = []
        if ar.model_history_messages > 0:
            history_context_lines = history[-ar.model_history_messages :]
        history_context = (
            "\n".join(history_context_lines)
            if history_context_lines
            else "(disabled or no additional history)"
        )

        logger.info(
            "enhance-mode | model_choice | 开始判定 | "
            f"origin={origin} trigger={trigger_reason} stack_size={len(messages)} "
            f"history={len(history_context_lines)}"
        )

        provider = self.context.get_using_provider(event.unified_msg_origin)
        if not provider or not isinstance(provider, Provider):
            logger.error("enhance-mode | 未找到可用提供商，无法执行模型选择触发")
            return False

        persona_name, persona_mask = await self._resolve_persona_mask(event)
        prompt_tmpl = ar.model_choice_prompt
        try:
            judge_prompt = prompt_tmpl.format(
                stack_size=len(messages),
                messages="\n".join(messages),
                history_count=len(history_context_lines),
                history_context=history_context,
                persona_name=persona_name,
                persona_mask=persona_mask,
            )
        except Exception:
            judge_prompt = (
                f"{prompt_tmpl}\n\n"
                f"人格面具({persona_name}):\n{persona_mask}\n\n"
                f"最近消息:\n{chr(10).join(messages)}\n\n"
                f"额外历史上下文({len(history_context_lines)}):\n{history_context}\n\n"
                "请仅输出 REPLY 或 SKIP。"
            )

        try:
            judge_resp = await asyncio.wait_for(
                provider.text_chat(
                    prompt=judge_prompt,
                    session_id=uuid.uuid4().hex,
                    persist=False,
                ),
                timeout=cfg.global_settings.timeouts.model_choice_sec,
            )
        except asyncio.TimeoutError:
            logger.error("enhance-mode | 模型选择触发判定超时")
            return False
        except Exception as e:
            logger.error(f"enhance-mode | 模型选择触发判定失败: {e}")
            return False

        decision_raw = (judge_resp.completion_text or "").strip().upper()
        decision = decision_raw.split()[0] if decision_raw else ""
        if decision.startswith("REPLY"):
            logger.info(
                "enhance-mode | model_choice | 判定通过(REPLY) | "
                f"origin={origin} trigger={trigger_reason} persona={persona_name}"
            )
            return True
        if decision and not decision.startswith("SKIP"):
            logger.info(
                "enhance-mode | model_choice | 判定拒绝(非标准输出按 SKIP) | "
                f"origin={origin} trigger={trigger_reason} output={decision_raw}"
            )
            return False
        logger.info(
            "enhance-mode | model_choice | 判定拒绝(SKIP) | "
            f"origin={origin} trigger={trigger_reason} persona={persona_name}"
        )
        return False

    async def _need_active_reply_model_choice(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        origin = event.unified_msg_origin
        self._touch_origin(origin, cfg)

        ar = cfg.active_reply
        text = (event.message_str or "").strip() or "[Empty]"
        nickname = event.message_obj.sender.nickname
        sender_id = event.get_sender_id()
        stack = self.runtime.active_reply_stacks[origin]
        history = self.runtime.model_choice_histories[origin]

        stack.append(f"[{nickname}/{sender_id}]: {text}")
        history_line = (
            f"[{nickname}/{sender_id}/"
            f"{datetime.datetime.now().strftime('%H:%M:%S')}]: {text}"
        )
        history.append(history_line)

        history_limit = max(
            60,
            ar.model_stack_size * 6,
            ar.model_history_messages * 6,
        )
        if len(history) > history_limit:
            del history[:-history_limit]

        logger.info(
            "enhance-mode | model_choice | 栈填充 | "
            f"origin={origin} progress={len(stack)}/{ar.model_stack_size} "
            f"sender={sender_id}"
        )

        if len(stack) < ar.model_stack_size:
            return False

        messages = stack[-ar.model_stack_size :]
        stack.clear()
        return await self._judge_model_choice(
            event,
            cfg,
            origin,
            messages,
            trigger_reason="stack_full",
        )

    async def _get_image_caption(
        self,
        image_url: str,
        provider_id: str,
        prompt: str,
        timeout_sec: float,
    ) -> str:
        if not provider_id:
            provider = self.context.get_using_provider()
        else:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise Exception(f"没有找到 ID 为 {provider_id} 的提供商")

        if not isinstance(provider, Provider):
            raise Exception(f"提供商类型错误({type(provider)})，无法获取图片描述")

        response = await asyncio.wait_for(
            provider.text_chat(
                prompt=prompt,
                session_id=uuid.uuid4().hex,
                image_urls=[image_url],
                persist=False,
            ),
            timeout=timeout_sec,
        )
        return response.completion_text

    async def _need_active_reply(
        self, event: AstrMessageEvent, cfg: PluginConfig
    ) -> bool:
        if not self._allow_active_reply(event, cfg):
            return False

        ar = cfg.active_reply
        if ar.mode == "model_choice":
            return await self._need_active_reply_model_choice(event, cfg)
        return random.random() < ar.possibility

    @filter.on_llm_request()
    async def inject_role(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        cfg = self._cfg()
        if not cfg.group_features.role_display:
            return

        base_cfg = self.context.get_config(umo=event.unified_msg_origin)
        if not base_cfg.get("identifier"):
            return

        role = "admin" if event.is_admin() else "member"
        role_line = f", Role: {role}"

        for part in req.extra_user_content_parts:
            if isinstance(part, TextPart) and "<system_reminder>" in part.text:
                if "Nickname: " in part.text and role_line not in part.text:
                    nickname_idx = part.text.index("Nickname: ")
                    rest = part.text[nickname_idx:]
                    newline_idx = rest.find("\n")
                    if newline_idx != -1:
                        insert_pos = nickname_idx + newline_idx
                    else:
                        close_idx = part.text.find("</system_reminder>")
                        insert_pos = close_idx if close_idx != -1 else len(part.text)
                    part.text = (
                        part.text[:insert_pos] + role_line + part.text[insert_pos:]
                    )
                return

        reminder = f"<system_reminder>Role: {role}</system_reminder>"
        req.extra_user_content_parts.append(TextPart(text=reminder))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        cfg = self._cfg()
        if not cfg.group_history_enabled and not cfg.active_reply_enabled:
            return

        has_content = any(
            isinstance(comp, (Plain, Image, Reply))
            for comp in event.message_obj.message
        )
        if not has_content:
            return

        need_active = await self._need_active_reply(event, cfg)

        if cfg.group_history_enabled:
            try:
                await self._record_message(event, cfg)
            except Exception as e:
                logger.error(f"enhance-mode | record message error: {e}")

        if need_active:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.error("enhance-mode | 未找到任何 LLM 提供商，无法主动回复")
                return
            try:
                if hasattr(event, "set_extra"):
                    event.set_extra("_enhance_active_reply_triggered", True)
                    event.set_extra("_enhance_active_reply_mode", cfg.active_reply.mode)

                session_curr_cid = (
                    await self.context.conversation_manager.get_curr_conversation_id(
                        event.unified_msg_origin,
                    )
                )
                if not session_curr_cid:
                    logger.error(
                        "enhance-mode | 当前未处于对话状态，无法主动回复，"
                        "请使用 /switch 或 /new 创建一个会话。"
                    )
                    return

                conv = await self.context.conversation_manager.get_conversation(
                    event.unified_msg_origin,
                    session_curr_cid,
                )
                if not conv:
                    logger.error("enhance-mode | 未找到对话，无法主动回复")
                    return

                yield event.request_llm(
                    prompt=event.message_str,
                    session_id=event.session_id,
                    conversation=conv,
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error(f"enhance-mode | 主动回复失败: {e}")

    async def _record_message(self, event: AstrMessageEvent, cfg: PluginConfig) -> None:
        history_cfg = cfg.group_history

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        nickname = event.message_obj.sender.nickname
        msg_id = event.message_obj.message_id

        if history_cfg.include_sender_id and history_cfg.include_role_tag:
            sender_id = event.get_sender_id()
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{sender_id}/{datetime_str}]{role_tag} #msg{msg_id}:"
        elif history_cfg.include_sender_id:
            sender_id = event.get_sender_id()
            header = f"[{nickname}/{sender_id}/{datetime_str}] #msg{msg_id}:"
        elif history_cfg.include_role_tag:
            role_tag = "(admin)" if event.is_admin() else "(member)"
            header = f"[{nickname}/{datetime_str}]{role_tag} #msg{msg_id}:"
        else:
            header = f"[{nickname}/{datetime_str}] #msg{msg_id}:"

        parts = [header]
        for comp in event.get_messages():
            if isinstance(comp, Reply):
                quote_nick = comp.sender_nickname or "Unknown"
                quote_text = (comp.message_str or "").strip() or "..."
                parts.append(f" [Quote {quote_nick}: {quote_text}]")
            elif isinstance(comp, Plain):
                parts.append(f" {comp.text}")
            elif isinstance(comp, Image):
                if history_cfg.image_caption:
                    try:
                        url = comp.url if comp.url else comp.file
                        if not url:
                            raise Exception("图片 URL 为空")
                        caption = await self._get_image_caption(
                            url,
                            history_cfg.image_caption_provider_id,
                            history_cfg.image_caption_prompt,
                            cfg.global_settings.timeouts.image_caption_sec,
                        )
                        parts.append(f" [Image: {caption}]")
                    except Exception as e:
                        logger.error(f"enhance-mode | 获取图片描述失败: {e}")
                        parts.append(" [Image]")
                else:
                    parts.append(" [Image]")
            elif isinstance(comp, At):
                parts.append(f" [At: {comp.name}]")

        final_message = "".join(parts)
        logger.debug(f"enhance-mode | {event.unified_msg_origin} | {final_message}")

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > history_cfg.max_messages:
            chats.pop(0)

    @filter.on_llm_request()
    async def inject_group_context(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return

        self._touch_origin(event.unified_msg_origin, cfg)
        bounded_chats = bounded_chat_history_text(
            self.runtime.session_chats[event.unified_msg_origin]
        )
        interaction_instructions = build_interaction_instructions(
            cfg.group_features.mention_parse,
            cfg.group_history.include_sender_id,
        )

        if (
            cfg.group_features.react_mode_enable
            and event.get_message_type() == MessageType.GROUP_MESSAGE
        ):
            is_active_triggered = event.get_extra(
                "_enhance_active_reply_triggered", False
            )
            active_mode = event.get_extra("_enhance_active_reply_mode", "")
            if is_active_triggered and active_mode == "model_choice":
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats}\n\n"
                    "You decided to actively join this conversation because some recent messages are worth replying to.\n"
                    "Choose the message(s) you want to respond to from the chat history above, "
                    "and compose a natural reply. Quote the message you choose in most cases.\n"
                    "Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            else:
                prompt = req.prompt
                req.prompt = (
                    f"You are now in a chatroom. The chat history is as follows:\n{bounded_chats}\n\n"
                    f"Now, a new message is coming: `{prompt}`. "
                    "Please react to it. Your entire output is your reply to this message. "
                    "Quote the message which is coming in most cases. "
                    "Only output your response and do not output any other information. "
                    "You MUST use the SAME language as the chatroom is using."
                    f"{interaction_instructions}"
                )
            req.contexts = []
        else:
            req.system_prompt += (
                "You are now in a chatroom. The chat history is as follows: \n"
            )
            req.system_prompt += bounded_chats
            req.system_prompt += interaction_instructions

    @filter.on_decorating_result()
    async def parse_tags(self, event: AstrMessageEvent) -> None:
        result = event.get_result()
        if not result or not result.chain:
            return

        # 全局拦截：模型输出 <refuse/> 时直接清空结果链，阻止后续发送到平台。
        if chain_has_refuse_tag(result.chain):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，已取消发送 | "
                f"origin={event.unified_msg_origin}"
            )
            if hasattr(event, "set_extra"):
                event.set_extra("_enhance_refused_reply", True)
            result.chain = []
            return

        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        transformed = transform_result_chain(
            result.chain,
            parse_mention=self._cfg().group_features.mention_parse,
        )
        if transformed is None:
            return

        result.chain = transformed

    @filter.on_llm_response()
    async def record_bot_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        cfg = self._cfg()
        if not cfg.group_history_enabled:
            return
        if event.unified_msg_origin not in self.runtime.session_chats:
            return
        if not resp.completion_text:
            return

        if has_refuse_tag(resp.completion_text):
            logger.info(
                "enhance-mode | 检测到 <refuse/>，跳过机器人回复历史记录 | "
                f"origin={event.unified_msg_origin}"
            )
            return

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        text = clean_response_text_for_history(resp.completion_text)
        final_message = f"[You/{datetime_str}]: {text}"

        logger.debug(
            f"enhance-mode | recorded AI response: "
            f"{event.unified_msg_origin} | {final_message}"
        )

        self._touch_origin(event.unified_msg_origin, cfg)
        chats = self.runtime.session_chats[event.unified_msg_origin]
        chats.append(final_message)
        if len(chats) > cfg.group_history.max_messages:
            chats.pop(0)

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        clean_session = event.get_extra("_clean_ltm_session", False)
        if not clean_session:
            return
        self.runtime.cleanup_origin(event.unified_msg_origin)
