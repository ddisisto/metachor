# metachor/ensemble.py
from metachor.types import Phase, ResourceConstraints, Message
from metachor.voice import Voice
import asyncio
import time
import logging

log = logging.getLogger("metachor")

class PhaseTimeoutError(Exception):
    """Raised when a phase exceeds its time budget."""
    pass

class Ensemble:
    # Updated phase contexts to include initialization
    PHASE_CONTEXTS = {
        Phase.INITIALIZATION: "Discuss how we can best work together as an ensemble of models",
        Phase.USER_ANALYSIS: "Analyze the key aspects and requirements of this user request",
        Phase.RESPONSE_PLANNING: "Based on our analysis, create a structured plan for the response",
        Phase.RESPONSE_DRAFTING: "Following our plan, generate a coherent response",
        Phase.RESPONSE_REFINING: "Refine and improve the drafted response while maintaining coherence"
    }
    
    # Updated phase budgets to include initialization
    PHASE_BUDGETS = {
        Phase.INITIALIZATION: {"time": 0.1, "tokens": 0.1},  # 10% each for init
        Phase.USER_ANALYSIS: {"time": 0.2, "tokens": 0.2},   # 20% each for analysis
        Phase.RESPONSE_PLANNING: {"time": 0.2, "tokens": 0.2}, # 20% each for planning
        Phase.RESPONSE_DRAFTING: {"time": 0.4, "tokens": 0.4}, # 40% each for drafting
        Phase.RESPONSE_REFINING: {"time": 0.1, "tokens": 0.1}  # 10% each for refinement
    }
    
    def __init__(self, voices: list[Voice], system_prompt: str | None = None):
        self.voices = voices
        self.system_prompt = system_prompt
        self.conversation_history: list[Message] = []
        self._start_time = None
        self._total_tokens = 0
        self._phase_tokens: dict[Phase, int] = {phase: 0 for phase in Phase}
        self._phase_responses: dict[Phase, list[Message]] = {phase: [] for phase in Phase}

    async def _run_phase_with_timeout(
        self, 
        phase: Phase, 
        content: str,
        constraints: ResourceConstraints,
        context: str | None = None
    ) -> list[Message]:
        """Run a phase with timeout and token budget, preserving context."""
        time_budget = constraints.max_time * self.PHASE_BUDGETS[phase]["time"]
        token_budget = int(constraints.max_tokens * self.PHASE_BUDGETS[phase]["tokens"])
        
        log.info(f"\n⏱️ Phase {phase.value} budget - Time: {time_budget:.1f}s, Tokens: {token_budget}")
        
        # Build contextual prompt
        phase_prompt = (
            f"{self.PHASE_CONTEXTS[phase]}:\n\n"
            f"{context + '\n\n' if context else ''}"
            f"{content}"
        )
        
        try:
            tasks = [
                asyncio.create_task(
                    voice.send(
                        content=phase_prompt,
                        to_model=self._get_next_voice(voice).model_id,
                        phase=phase,
                        max_tokens=token_budget // len(self.voices)
                    )
                )
                for voice in self.voices
            ]
            
            # Wait for all tasks with timeout
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=time_budget
            )
            
            # Store responses in phase history
            self._phase_responses[phase].extend(responses)
            
            # Update token counts
            phase_tokens = sum(r.tokens_used for r in responses)
            self._phase_tokens[phase] = phase_tokens
            self._total_tokens += phase_tokens
            
            log.info(f"✅ Phase {phase.value} completed - {phase_tokens} tokens used")
            return responses
            
        except asyncio.TimeoutError:
            log.warning(f"⚠️ Phase {phase.value} timed out after {time_budget:.1f}s")
            # Return any completed responses before timeout
            completed_responses = [t.result() for t in tasks if t.done()]
            if completed_responses:
                self._phase_responses[phase].extend(completed_responses)
            return completed_responses

    async def send(
        self, 
        user_input: str, 
        constraints: ResourceConstraints,
        include_initialization: bool = True
    ) -> str:
        """Process user request through multi-model collaboration."""
        self._start_time = time.time()
        self._total_tokens = 0
        self._phase_responses = {phase: [] for phase in Phase}
        
        log.info(f"\n📥 Processing user input: {user_input}")
        log.info(f"Constraints: {constraints}")

        try:
            # Optional initialization phase
            if include_initialization:
                init_responses = await self._run_phase_with_timeout(
                    Phase.INITIALIZATION,
                    "How should we work together to best serve users? What are our unique strengths?",
                    constraints
                )
                init_summary = self._summarize_responses(init_responses)
                log.debug(f"Initialization summary: {init_summary}")

            # Continue with regular phases...
            analysis_responses = await self._run_phase_with_timeout(
                Phase.USER_ANALYSIS,
                f"User request: {user_input}",
                constraints
            )
            
            analysis_summary = self._summarize_responses(analysis_responses)
            
            plan_responses = await self._run_phase_with_timeout(
                Phase.RESPONSE_PLANNING,
                "How should we structure the response?",
                constraints,
                context=f"Analysis summary:\n{analysis_summary}"
            )
            
            plan_summary = self._summarize_responses(plan_responses)
            
            draft_responses = await self._run_phase_with_timeout(
                Phase.RESPONSE_DRAFTING,
                f"Generate response for: {user_input}",
                constraints,
                context=f"Analysis:\n{analysis_summary}\n\nPlan:\n{plan_summary}"
            )
            
            return self._format_final_response(draft_responses)

        except Exception as e:
            log.error(f"❌ Error during response generation: {str(e)}")
            return self._format_final_response(self._get_all_responses())   

    def _summarize_responses(self, responses: list[Message]) -> str:
        """Extract key points from responses into a concise summary."""
        if not responses:
            return "No responses available"
            
        # Combine all response content
        combined = "\n\n".join(r.content for r in responses)
        
        # Simple extraction of key points (could be made smarter)
        key_points = []
        for line in combined.split("\n"):
            line = line.strip()
            if line and (line.startswith("- ") or line.startswith("• ") or 
                        line.startswith("* ") or line.startswith("1.")):
                key_points.append(line)
        
        return "\n".join(key_points) if key_points else combined[:500]  # Limit summary size

    def _get_all_responses(self) -> list[Message]:
        """Get all responses from all phases."""
        all_responses = []
        for phase in Phase:
            all_responses.extend(self._phase_responses[phase])
        return all_responses

    def _format_final_response(self, responses: list[Message]) -> str:
        """Format the final response with all available content."""
        if not responses:
            # Try to salvage content from any phase
            responses = self._get_all_responses()
            if not responses:
                return "No response generated within resource constraints."
        
        # Combine responses meaningfully
        contents = []
        seen_content = set()  # Track unique content
        
        for r in responses:
            content = r.content.strip()
            # Simple deduplication
            if content and content not in seen_content:
                contents.append(content)
                seen_content.add(content)
        
        response = "\n\n".join(contents)
        
        # Add metadata
        metadata = [
            f"\n---",
            f"Time usage: {time.time() - self._start_time:.2f}s",
            f"Total tokens: {self._total_tokens}",
            "Token usage by phase:"
        ]
        for phase, tokens in self._phase_tokens.items():
            if tokens > 0:
                metadata.append(f"- {phase.value}: {tokens} tokens")
        
        return f"{response.strip()}\n{''.join(metadata)}"

    def _get_next_voice(self, current_voice: Voice) -> Voice:
        """Get the next voice in rotation, with error handling.
        
        Args:
            current_voice: The current Voice instance
            
        Returns:
            The next Voice instance in the rotation
            
        Raises:
            ValueError: If current_voice is not in the ensemble
        """
        if not self.voices:
            raise ValueError("No voices available in ensemble")
            
        try:
            current_idx = self.voices.index(current_voice)
        except ValueError:
            log.error(f"Voice {current_voice.model_id} not found in ensemble")
            # Fallback to first voice if current not found
            return self.voices[0]
            
        next_idx = (current_idx + 1) % len(self.voices)
        next_voice = self.voices[next_idx]
        
        log.debug(f"Rotating from {current_voice.model_id} to {next_voice.model_id}")
        return next_voice
