# metachor/voice.py
from typing import Optional
import httpx
import asyncio
from metachor.types import Message, Phase

class Voice:
    """Represents a single LLM in the ensemble, handling its interactions and state."""
    
    def __init__(self, 
                 model_id: str,
                 api_key: str,
                 system_prompt: str | None = None,
                 max_tokens: int = 1000):
        self.model_id = model_id
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.conversation_history: list[Message] = []
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/yourusername/metachor",  # Required by OpenRouter
                "X-Title": "metachor"  # Optional but good practice
            }
        )

    async def send(self, 
                  content: str,
                  to_model: str,
                  phase: Phase,
                  context: list[Message] | None = None) -> Message:
        """Generate a response to the given content."""
        messages = self._prepare_messages(content, context)
        
        try:
            async with self.client:
                response = await self.client.post(
                    "/chat/completions",
                    json={
                        "model": self.model_id,
                        "messages": messages,
                        "max_tokens": self.max_tokens
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract content and token usage
                response_content = data["choices"][0]["message"]["content"]
                tokens_used = data["usage"]["total_tokens"]
                
                return Message(
                    content=response_content,
                    tokens_used=tokens_used,
                    from_model=self.model_id,
                    to_model=to_model,
                    phase=phase
                )
                
        except httpx.HTTPError as e:
            # In production we'd want more sophisticated error handling
            raise RuntimeError(f"API call failed: {e}")

    def _prepare_messages(self, 
                         content: str, 
                         context: list[Message] | None = None) -> list[dict]:
        """Prepare messages for the API call."""
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
            
        if context:
            for msg in context:
                messages.append({
                    "role": "assistant" if msg.from_model == self.model_id else "user",
                    "content": f"[{msg.from_model} → {msg.to_model}] {msg.content}"
                })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages

    async def forget_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()