# metachor/ensemble.py
from metachor.types import Phase, ResourceConstraints, Message
from metachor.voice import Voice
import asyncio
import time

import logging

log = logging.getLogger("metachor")

class Ensemble:
    """A collaborative system of multiple LLMs working together to generate responses."""
    
    def __init__(self, voices: list[Voice], system_prompt: str | None = None):
        """Initialize collaborative ensemble with multiple LLMs.
        
        Args:
            voices: List of Voice instances representing different LLMs
            system_prompt: Optional base system prompt for all models
        """
        self.voices = voices
        self.system_prompt = system_prompt
        self.conversation_history: list[Message] = []

    async def initialize_meta_discussion(self, iterations: int = 3) -> None:
        """Initial phase where models discuss their roles and communication protocol."""
        log.info(f"\n🚀 Starting meta-discussion with {len(self.voices)} models")
        seed_question = "How should we work together to best serve users? What are our unique strengths?"
        
        for i in range(iterations):
            log.info(f"\n📝 Meta-discussion iteration {i+1}/{iterations}")
            for voice in self.voices:
                next_voice = self._get_next_voice(voice)
                response = await voice.send(
                    content=seed_question,
                    to_model=next_voice.model_id,
                    context=self.conversation_history,
                    phase=Phase.INITIALIZATION
                )
                self.conversation_history.append(response)
                log.debug(f"Meta-discussion history length: {len(self.conversation_history)}")

    async def send(self, user_input: str, constraints: ResourceConstraints) -> str:
        """Process user request through multi-model collaboration."""
        start_time = time.time()
        tokens_used = 0
        iterations = 0
        
        log.info(f"\n📥 Processing user input: {user_input}")
        log.info(f"Constraints: {constraints}")

        # Phase 1 & 2: Analysis and Planning
        log.info("\n🔍 Starting analysis phase")
        await self._collaborate_phase(
            phase=Phase.USER_ANALYSIS,
            context=f"User request: {user_input}\nConstraints: {constraints}",
            iterations=2
        )
        
        log.info("\n📋 Starting planning phase")
        await self._collaborate_phase(
            phase=Phase.RESPONSE_PLANNING,
            context="Based on our analysis, how should we structure the response?",
            iterations=2
        )

        # Phase 3: Iterative response development
        log.info("\n✍️ Starting response generation")
        draft_response = ""
        
        while (iterations < constraints.max_iterations and 
               tokens_used < constraints.max_tokens and
               time.time() - start_time < constraints.max_time):
            
            phase = Phase.RESPONSE_DRAFTING
            if tokens_used > constraints.max_tokens * 0.7:
                phase = Phase.RESPONSE_REFINING
                log.info("\n🔄 Switching to refinement phase")
            
            log.info(f"\n📝 Generation iteration {iterations + 1}/{constraints.max_iterations}")
            log.debug(f"Current tokens: {tokens_used}/{constraints.max_tokens}")
            log.debug(f"Time elapsed: {time.time() - start_time:.2f}s/{constraints.max_time}s")
            
            responses = await asyncio.gather(*[
                voice.send(
                    content=draft_response if draft_response else user_input,
                    to_model=self._get_next_voice(voice).model_id,
                    context=self.conversation_history,
                    phase=phase
                )
                for voice in self.voices
            ])
            
            for response in responses:
                draft_response = self._integrate_response(draft_response, response)
                tokens_used += response.tokens_used
                iterations += 1
                
                if self._is_response_complete(draft_response, constraints):
                    log.info("\n✅ Response complete")
                    break

        log.info(f"\n🏁 Generation complete - {iterations} iterations, {tokens_used} tokens, {time.time() - start_time:.2f}s")
        return self._format_final_response(draft_response)

    async def _collaborate_phase(self, phase: Phase, context: str, iterations: int) -> None:
        """Run a collaboration phase between models."""
        for _ in range(iterations):
            for voice in self.voices:
                next_voice = self._get_next_voice(voice)
                response = await voice.send(
                    content=context,
                    to_model=next_voice.model_id,
                    context=self.conversation_history,
                    phase=phase
                )
                self.conversation_history.append(response)

    def _get_next_voice(self, current_voice: Voice) -> Voice:
        """Get the next voice in rotation."""
        current_idx = self.voices.index(current_voice)
        next_idx = (current_idx + 1) % len(self.voices)
        return self.voices[next_idx]
    
    def _integrate_response(self, current: str, new_response: Message) -> str:
        """Integrate new model response with current draft."""
        # For now, just append - we'll make this smarter later
        if not current:
            return new_response.content
        return f"{current}\n\n{new_response.content}"
    
    def _is_response_complete(self, response: str, constraints: ResourceConstraints) -> bool:
        """Check if response meets completion criteria."""
        # Simple implementation for now - just check token count
        estimated_tokens = len(response.split())  # Very naive
        return estimated_tokens >= constraints.max_tokens * 0.9
    
    def _format_final_response(self, response: str) -> str:
        """Format the final response for user consumption."""
        # For now, just return as is - we can add formatting later
        return response
