import datetime
import random
import re
import traceback
import uuid
from collections import defaultdict

from astrbot.api import logger, star
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Plain
from astrbot.api.platform import MessageType
from astrbot.api.provider import LLMResponse, Provider, ProviderRequest
from astrbot.core.agent.message import TextPart


class Main(star.Star):
    def __init__(self, context: star.Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        self.context = context
        self.config = config or {}
        self.session_chats: dict[str, list[str]] = defaultdict(list)

    def _react_mode_cfg(self):
        rm = self.config.get("react_mode", {})
        return {
            "enable": rm.get("enable", False),
        }

    def _group_context_cfg(self):
        gc = self.config.get("group_context", {})
        react_mode_enable = self._react_mode_cfg()["enable"]
        return {
            "enable": gc.get("enable", False) and react_mode_enable,
            "max_messages": gc.get("max_messages", 300),
            "include_sender_id": gc.get("include_sender_id", True),
            "include_role_tag": gc.get("include_role_tag", True),
            "image_caption": gc.get("image_caption", False)
            and bool(gc.get("image_caption_provider_id")),
            "image_caption_provider_id": gc.get("image_caption_provider_id", ""),
            "image_caption_prompt": gc.get(
                "image_caption_prompt", "Describe this image in one sentence."
            ),
        }

    def _active_reply_cfg(self):
        ar = self.config.get("active_reply", {})
        react_mode_enable = self._react_mode_cfg()["enable"]
        whitelist_str = ar.get("whitelist", "")
        whitelist = (
            [w.strip() for w in whitelist_str.split(",") if w.strip()]
            if whitelist_str
            else []
        )
        return {
            "enable": ar.get("enable", False) and react_mode_enable,
            "possibility": ar.get("possibility", 0.1),
            "whitelist": whitelist,
        }

    async def _get_image_caption(
        self, image_url: str, provider_id: str, prompt: str
    ) -> str:
        if not provider_id:
            provider = self.context.get_using_provider()
        else:
            provider = self.context.get_provider_by_id(provider_id)
            if not provider:
                raise Exception(f"没有找到 ID 为 {provider_id} 的提供商")
        if not isinstance(provider, Provider):
            raise Exception(f"提供商类型错误({type(provider)})，无法获取图片描述")
        response = await provider.text_chat(
            prompt=prompt,
            session_id=uuid.uuid4().hex,
            image_urls=[image_url],
            persist=False,
        )
        return response.completion_text

    def _need_active_reply(self, event: AstrMessageEvent) -> bool:
        ar = self._active_reply_cfg()
        if not ar["enable"]:
            return False
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return False
        if event.is_at_or_wake_command:
            return False
        if ar["whitelist"] and (
            event.unified_msg_origin not in ar["whitelist"]
            and (event.get_group_id() and event.get_group_id() not in ar["whitelist"])
        ):
            return False
        return random.random() < ar["possibility"]

    @filter.on_llm_request()
    async def inject_role(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """Inject user role into the existing system_reminder block."""
        if not self.config.get("role_display", True):
            return

        cfg = self.context.get_config(umo=event.unified_msg_origin)
        if not cfg.get("identifier"):
            return

        role = "admin" if event.is_admin() else "member"
        role_line = f", Role: {role}"

        # Find the existing system_reminder TextPart and inject role into it
        for part in req.extra_user_content_parts:
            if isinstance(part, TextPart) and "<system_reminder>" in part.text:
                # Insert role after the Nickname line
                if "Nickname: " in part.text and role_line not in part.text:
                    # Find the end of the Nickname value (next newline or </system_reminder>)
                    nickname_idx = part.text.index("Nickname: ")
                    # Find the end of this line
                    rest = part.text[nickname_idx:]
                    newline_idx = rest.find("\n")
                    if newline_idx != -1:
                        insert_pos = nickname_idx + newline_idx
                    else:
                        insert_pos = part.text.index("</system_reminder>")
                    part.text = (
                        part.text[:insert_pos] + role_line + part.text[insert_pos:]
                    )
                return

        # Fallback: no existing system_reminder found, add a standalone one
        reminder = f"<system_reminder>Role: {role}</system_reminder>"
        req.extra_user_content_parts.append(TextPart(text=reminder))

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_group_message(self, event: AstrMessageEvent):
        """Record group messages and handle active reply."""
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        gc = self._group_context_cfg()
        if not gc["enable"] and not self._active_reply_cfg()["enable"]:
            return

        has_image_or_plain = any(
            isinstance(comp, (Plain, Image)) for comp in event.message_obj.message
        )
        if not has_image_or_plain:
            return

        need_active = self._need_active_reply(event)

        if gc["enable"]:
            try:
                await self._record_message(event, gc)
            except Exception as e:
                logger.error(f"enhance-mode | record message error: {e}")

        if need_active:
            provider = self.context.get_using_provider(event.unified_msg_origin)
            if not provider:
                logger.error("enhance-mode | 未找到任何 LLM 提供商，无法主动回复")
                return
            try:
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

    async def _record_message(self, event: AstrMessageEvent, gc: dict):
        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        nickname = event.message_obj.sender.nickname

        if gc["include_sender_id"] and gc["include_role_tag"]:
            sender_id = event.get_sender_id()
            role_tag = "(admin)" if event.is_admin() else "(member)"
            parts = [f"[{nickname}/{sender_id}/{datetime_str}]{role_tag}: "]
        elif gc["include_sender_id"]:
            sender_id = event.get_sender_id()
            parts = [f"[{nickname}/{sender_id}/{datetime_str}]: "]
        elif gc["include_role_tag"]:
            role_tag = "(admin)" if event.is_admin() else "(member)"
            parts = [f"[{nickname}/{datetime_str}]{role_tag}: "]
        else:
            parts = [f"[{nickname}/{datetime_str}]: "]

        for comp in event.get_messages():
            if isinstance(comp, Plain):
                parts.append(f" {comp.text}")
            elif isinstance(comp, Image):
                if gc["image_caption"]:
                    try:
                        url = comp.url if comp.url else comp.file
                        if not url:
                            raise Exception("图片 URL 为空")
                        caption = await self._get_image_caption(
                            url,
                            gc["image_caption_provider_id"],
                            gc["image_caption_prompt"],
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
        self.session_chats[event.unified_msg_origin].append(final_message)
        if len(self.session_chats[event.unified_msg_origin]) > gc["max_messages"]:
            self.session_chats[event.unified_msg_origin].pop(0)

    @filter.on_llm_request()
    async def inject_group_context(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        """Inject recorded group chat history into system prompt."""
        gc = self._group_context_cfg()
        react_mode = self._react_mode_cfg()
        if not gc["enable"]:
            return
        if event.unified_msg_origin not in self.session_chats:
            return

        chats_str = "\n---\n".join(self.session_chats[event.unified_msg_origin])

        if (
            react_mode["enable"]
            and event.get_message_type() == MessageType.GROUP_MESSAGE
        ):
            prompt = req.prompt
            req.prompt = (
                f"You are now in a chatroom. The chat history is as follows:\n{chats_str}"
                f"\nNow, a new message is coming: `{prompt}`. "
                "Please react to it. Only output your response and do not output any other information. "
                "You MUST use the SAME language as the chatroom is using."
            )
            req.contexts = []
        else:
            req.system_prompt += (
                "You are now in a chatroom. The chat history is as follows: \n"
            )
            req.system_prompt += chats_str

        if self.config.get("mention_parse", True) and gc["include_sender_id"]:
            req.system_prompt += (
                "\n\n## Mention\n"
                'When you want to mention/@ a user in your reply, use the format: <mention id="user_id">.\n'
                'For example: <mention id="123456"> Hello!\n'
                "You can mention multiple users in one message. "
                "The user_id can be found in the chat history format [nickname/user_id/time].\n"
                "Do NOT use this format for yourself."
            )

    _MENTION_RE = re.compile(r'<mention\s+id="([^"]+)"\s*/?>')

    @filter.on_decorating_result()
    async def parse_mentions(self, event: AstrMessageEvent) -> None:
        """Parse <mention id="xxx"> tags in LLM output and replace with At components.

        Note: This hook is NOT called for STREAMING_RESULT (the pipeline returns
        early before hooks run). For STREAMING_FINISH the text has already been
        sent. Therefore mention parsing only works with streaming disabled.
        """
        if not self.config.get("mention_parse", True):
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        result = event.get_result()
        if not result or not result.chain:
            return

        # Check if any Plain component contains mention tags
        has_mention = any(
            isinstance(comp, Plain) and self._MENTION_RE.search(comp.text)
            for comp in result.chain
        )
        if not has_mention:
            return

        new_chain = []
        for comp in result.chain:
            if not isinstance(comp, Plain) or not self._MENTION_RE.search(comp.text):
                new_chain.append(comp)
                continue
            parts = self._MENTION_RE.split(comp.text)
            # parts: [text, id, text, id, text, ...]
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    if part:
                        new_chain.append(Plain(text=part))
                else:
                    new_chain.append(At(qq=part))
        result.chain = new_chain

    @filter.on_llm_response()
    async def record_bot_response(
        self, event: AstrMessageEvent, resp: LLMResponse
    ) -> None:
        """Record bot response to group chat history."""
        gc = self._group_context_cfg()
        if not gc["enable"]:
            return
        if event.unified_msg_origin not in self.session_chats:
            return
        if not resp.completion_text:
            return

        datetime_str = datetime.datetime.now().strftime("%H:%M:%S")
        # Clean mention tags for history recording
        text = self._MENTION_RE.sub(r"[At: \1]", resp.completion_text)
        final_message = f"[You/{datetime_str}]: {text}"
        logger.debug(
            f"enhance-mode | recorded AI response: "
            f"{event.unified_msg_origin} | {final_message}"
        )
        self.session_chats[event.unified_msg_origin].append(final_message)
        if len(self.session_chats[event.unified_msg_origin]) > gc["max_messages"]:
            self.session_chats[event.unified_msg_origin].pop(0)

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent) -> None:
        """Clean up session chats when conversation is cleared."""
        gc = self._group_context_cfg()
        if not gc["enable"]:
            return
        clean_session = event.get_extra("_clean_ltm_session", False)
        if clean_session and event.unified_msg_origin in self.session_chats:
            del self.session_chats[event.unified_msg_origin]
