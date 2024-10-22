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
    """A collaborative system of multiple LLMs working together to generate responses."""
    
    # Phase resource allocation (percentages)
    PHASE_BUDGETS = {
        Phase.INITIALIZATION: {"time": 0.1, "tokens": 0.1},  # 10% each for init
        Phase.USER_ANALYSIS: {"time": 0.2, "tokens": 0.2},   # 20% each for analysis
        Phase.RESPONSE_PLANNING: {"time": 0.2, "tokens": 0.2}, # 20% each for planning
        Phase.RESPONSE_DRAFTING: {"time": 0.4, "tokens": 0.4}, # 40% each for drafting/refining
        Phase.RESPONSE_REFINING: {"time": 0.1, "tokens": 0.1}  # 10% each for final refinement
    }
    
    def __init__(self, voices: list[Voice], system_prompt: str | None = None):
        self.voices = voices
        self.system_prompt = system_prompt
        self.conversation_history: list[Message] = []
        self._start_time = None
        self._total_tokens = 0
        self._phase_tokens: dict[Phase, int] = {phase: 0 for phase in Phase}

    async def _run_phase_with_timeout(
        self, 
        phase: Phase, 
        content: str,
        constraints: ResourceConstraints
    ) -> list[Message]:
        """Run a phase with timeout and token budget."""
        # Calculate phase budgets
        time_budget = constraints.max_time * self.PHASE_BUDGETS[phase]["time"]
        token_budget = int(constraints.max_tokens * self.PHASE_BUDGETS[phase]["tokens"])
        
        log.info(f"\n⏱️ Phase {phase.value} budget - Time: {time_budget:.1f}s, Tokens: {token_budget}")
        
        try:
            # Create tasks for all voices
            tasks = [
                asyncio.create_task(
                    voice.send(
                        content=content,
                        to_model=self._get_next_voice(voice).model_id,
                        phase=phase,
                        max_tokens=token_budget // len(self.voices)  # Split token budget among voices
                    )
                )
                for voice in self.voices
            ]
            
            # Wait for all tasks with timeout
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=time_budget
            )
            
            # Update token counts
            phase_tokens = sum(r.tokens_used for r in responses)
            self._phase_tokens[phase] = phase_tokens
            self._total_tokens += phase_tokens
            
            log.info(f"✅ Phase {phase.value} completed - {phase_tokens} tokens used")
            return responses
            
        except asyncio.TimeoutError:
            log.warning(f"⚠️ Phase {phase.value} timed out after {time_budget:.1f}s")
            # Cancel any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise PhaseTimeoutError(f"Phase {phase.value} exceeded time budget")

    async def send(self, user_input: str, constraints: ResourceConstraints) -> str:
        """Process user request through multi-model collaboration."""
        self._start_time = time.time()
        self._total_tokens = 0
        draft_response = ""
        
        log.info(f"\n📥 Processing user input: {user_input}")
        log.info(f"Constraints: {constraints}")

        try:
            # Analysis phase
            analysis_responses = await self._collaborate_phase(
                phase=Phase.USER_ANALYSIS,
                content=f"User request: {user_input}. What aspects should we focus on?",
                constraints=constraints,
                iterations=2
            )
            
            remaining_time = constraints.max_time - (time.time() - self._start_time)
            remaining_tokens = constraints.max_tokens - self._total_tokens
            
            if remaining_time > 0 and remaining_tokens > 0:
                # Planning phase with remaining resources
                plan_constraints = ResourceConstraints(
                    max_tokens=remaining_tokens // 2,
                    max_iterations=constraints.max_iterations // 2,
                    max_time=remaining_time / 2
                )
                
                analysis_summary = self._integrate_responses(analysis_responses)
                plan_responses = await self._collaborate_phase(
                    phase=Phase.RESPONSE_PLANNING,
                    content=f"Based on this analysis:\n{analysis_summary}\nHow should we structure the response?",
                    constraints=plan_constraints,
                    iterations=1
                )
                
                # Final response generation with remaining resources
                remaining_time = constraints.max_time - (time.time() - self._start_time)
                remaining_tokens = constraints.max_tokens - self._total_tokens
                
                if remaining_time > 0 and remaining_tokens > 0:
                    draft_constraints = ResourceConstraints(
                        max_tokens=remaining_tokens,
                        max_iterations=max(1, constraints.max_iterations // 2),
                        max_time=remaining_time
                    )
                    
                    plan_summary = self._integrate_responses(plan_responses)
                    draft_responses = await self._collaborate_phase(
                        phase=Phase.RESPONSE_DRAFTING,
                        content=f"Following this plan:\n{plan_summary}\nGenerate response for: {user_input}",
                        constraints=draft_constraints,
                        iterations=1
                    )
                    
                    draft_response = self._integrate_responses(draft_responses)

        except Exception as e:
            log.error(f"❌ Error during response generation: {str(e)}")
            if not draft_response:
                draft_response = f"Error during response generation: {str(e)}"

        return self._format_final_response(draft_response)

    async def initialize_meta_discussion(self, iterations: int = 1) -> None:
        """Initial phase where models discuss their roles and communication protocol."""
        log.info(f"\n🚀 Starting meta-discussion with {len(self.voices)} models")
        seed_question = "How should we work together to best serve users? What are our unique strengths?"
        
        try:
            # Create minimal constraints for meta-discussion
            meta_constraints = ResourceConstraints(
                max_tokens=200 * len(self.voices),  # 200 tokens per model
                max_iterations=iterations,
                max_time=5.0  # Short time limit for meta-discussion
            )
            
            responses = await self._run_phase_with_timeout(
                Phase.INITIALIZATION,
                seed_question,
                meta_constraints
            )
            
            # Store responses in conversation history
            self.conversation_history.extend(responses)
            
        except PhaseTimeoutError:
            log.warning("⚠️ Meta-discussion timed out, continuing with execution")
        except Exception as e:
            log.error(f"❌ Error during meta-discussion: {str(e)}")

    async def _collaborate_phase(
        self,
        phase: Phase,
        content: str,
        constraints: ResourceConstraints,
        iterations: int = 1
    ) -> list[Message]:
        """Run a collaboration phase between models with resource constraints."""
        all_responses = []
        
        try:
            for i in range(iterations):
                log.info(f"\n📝 Collaboration iteration {i+1}/{iterations} for phase {phase.value}")
                
                # Calculate remaining budget for this iteration
                remaining_time = constraints.max_time - (time.time() - self._start_time)
                remaining_tokens = constraints.max_tokens - self._total_tokens
                
                if remaining_time <= 0 or remaining_tokens <= 0:
                    log.warning(f"⚠️ Resources exhausted during {phase.value} iteration {i+1}")
                    break
                
                # Create constraints for this iteration
                iteration_constraints = ResourceConstraints(
                    max_tokens=remaining_tokens // (iterations - i),  # Divide remaining tokens
                    max_iterations=1,
                    max_time=remaining_time / (iterations - i)  # Divide remaining time
                )
                
                responses = await self._run_phase_with_timeout(
                    phase,
                    content,
                    iteration_constraints
                )
                
                all_responses.extend(responses)
                self.conversation_history.extend(responses)
                
                # Update content with responses for next iteration
                content = self._integrate_responses(responses)
                
        except PhaseTimeoutError:
            log.warning(f"⚠️ Phase {phase.value} timed out after {iterations} iterations")
        except Exception as e:
            log.error(f"❌ Error during {phase.value}: {str(e)}")
        
        return all_responses

    def _integrate_responses(self, responses: list[Message]) -> str:
        """Integrate multiple model responses into a coherent output."""
        if not responses:
            return ""
            
        # Simple concatenation for now - could be made smarter
        unique_contents = []
        for response in responses:
            content = response.content.strip()
            if content and content not in unique_contents:
                unique_contents.append(content)
                
        return "\n\n".join(unique_contents)

    def _format_final_response(self, response: str) -> str:
        """Format the final response with detailed metadata."""
        if not response:
            return "No response generated within resource constraints."
            
        # Add detailed metadata about resource usage
        elapsed_time = time.time() - self._start_time
        metadata = [
            f"\n---",
            f"Time usage: {elapsed_time:.2f}s",
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
        
    def _format_final_response(self, response: str) -> str:
        """Format the final response for user consumption."""
        if not response:
            return "No response generated within resource constraints."
            
        # Add metadata about resource usage
        metadata = f"\n\n---\nGenerated using {self._total_tokens} tokens in {time.time() - self._start_time:.2f}s"
        return f"{response.strip()}{metadata}"
