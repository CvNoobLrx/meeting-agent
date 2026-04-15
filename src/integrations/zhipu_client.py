import os
from typing import Any, List, Optional
import json
from zhipuai import ZhipuAI
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class ZhipuAIClient:
    """智谱 AI 客户端集成"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        self.base_url = base_url or os.getenv("ZHIPUAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
        self.model = os.getenv("ZHIPUAI_MODEL", "glm-4-flash")
        
        if not self.api_key:
            logger.warning("ZHIPUAI_API_KEY not found in environment variables.")

        self.client = None
        if self.api_key:
            self.client = ZhipuAI(api_key=self.api_key, base_url=self.base_url)

    async def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> str:
        """发送聊天请求"""
        if self.client is None:
            return "Error: ZHIPUAI_API_KEY is not configured"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=stream,
                **kwargs
            )
            if stream:
                return response # 返回生成器
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"ZhipuAI Chat Error: {e}")
            return f"Error: {str(e)}"

    async def generate_json(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> dict:
        """生成结构化 JSON (通常用于 Agent 提取信息)"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        content = await self.chat(messages, response_format={"type": "json_object"})
        try:
            return json.loads(content)
        except:
            return {"error": "Failed to parse JSON", "raw": content}

    async def chat_json(self, messages: List[dict], **kwargs) -> dict:
        """与 MiniMaxClient 兼容的方法：返回 JSON 字典。"""
        content = await self.chat(
            messages=messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

        if not isinstance(content, str):
            return {"error": "Non-string response", "raw": content}

        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()

        try:
            return json.loads(text)
        except Exception:
            logger.error("Failed to parse JSON response from ZhipuAI")
            return {"error": "Failed to parse JSON", "raw": content}
