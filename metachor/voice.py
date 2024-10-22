# metachor/voice.py
from typing import Optional
import httpx
from metachor.types import Phase, PHASE_CONTEXTS, Message
import logging

log = logging.getLogger("metachor")


class Voice:
    """Represents a single LLM in the ensemble, handling its interactions and state."""
    def __init__(self, 
                model_id: str,
                api_key: str,
                direct_prompt: str | None = None,
                collaborative_prompt: str | None = None,
                max_tokens: int = 1000):
        self.model_id = model_id
        self.api_key = api_key
        self.direct_prompt = direct_prompt
        self.collaborative_prompt = collaborative_prompt
        self.max_tokens = max_tokens
        self.conversation_history: list[Message] = []
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ddisisto/metachor",
            "X-Title": "metachor"
        }

    async def send(
        self,
        content: str,
        to_model: str,
        phase: Phase,
        max_tokens: Optional[int] = None,
        context: Optional[list[Message]] = None
    ) -> Message:
        """Generate a response to the given content with token budget."""
        messages = self._prepare_messages(content, context)
        
        log.debug(f"\n{'='*50}")
        log.info(f"🎯 {self.model_id} → {to_model} ({phase.value})")
        log.debug(f"Input content: {content[:200]}..." if len(content) > 200 else f"Input content: {content}")
        
        try:
            # Create a new client for each request
            async with httpx.AsyncClient(
                base_url="https://openrouter.ai/api/v1",
                headers=self.headers,
                timeout=30.0
            ) as client:
                request_body = {
                    "model": self.model_id,
                    "messages": messages,
                    "max_tokens": max_tokens or self.max_tokens
                }
                log.debug(f"Request: {request_body}")
                
                response = await client.post("/chat/completions", json=request_body)
                response.raise_for_status()
                data = response.json()
                
                response_content = data["choices"][0]["message"]["content"]
                tokens_used = data["usage"]["total_tokens"]
                prompt_tokens = data["usage"].get("prompt_tokens", 0)
                completion_tokens = data["usage"].get("completion_tokens", 0)
                
                log.info(f"📊 Tokens - Total: {tokens_used}, Prompt: {prompt_tokens}, Completion: {completion_tokens}")
                log.debug(f"Response content: {response_content[:200]}..." if len(response_content) > 200 else f"Response content: {response_content}")
                
                return Message(
                    content=response_content,
                    tokens_used=tokens_used,
                    from_model=self.model_id,
                    to_model=to_model,
                    phase=phase
                )
                
        except httpx.HTTPError as e:
            log.error(f"API call failed for {self.model_id}: {str(e)}")
            raise RuntimeError(f"API call failed: {e}")

    def _prepare_messages(self, content: str, context: list[Message] | None = None) -> list[dict]:
        """Prepare messages for the API call."""
        messages = []
        
        # Choose appropriate prompt based on mode
        if context:  # Collaborative mode
            current_phase = context[-1].phase if context else Phase.INITIALIZATION
            system_content = self.collaborative_prompt.format(
                phase=current_phase.value,
                phase_context=PHASE_CONTEXTS[current_phase]
            )
        else:  # Direct mode
            system_content = self.direct_prompt
            
        if system_content:
            messages.append({
                "role": "system",
                "content": system_content
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