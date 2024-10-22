# metachor/voice.py
from metachor.types import Message, Phase

class Voice:
    """Represents a single LLM in the ensemble, handling its interactions and state."""
    
    def __init__(self, 
                 model_id: str,
                 system_prompt: str | None = None,
                 max_tokens: int = 1000):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.conversation_history: list[Message] = []

    def send(self, 
            content: str,
            to_model: str,
            phase: Phase,
            context: list[Message] | None = None) -> Message:
        """Generate a response to the given content.
        
        Args:
            content: The message to respond to
            to_model: ID of the model this response is directed to
            phase: Current phase of the conversation
            context: Optional additional context messages
            
        Returns:
            Generated message with response
        """
        # TODO: Implement actual LLM call
        # For now, return dummy message
        return Message(
            content="Placeholder response",
            tokens_used=0,
            from_model=self.model_id,
            to_model=to_model,
            phase=phase
        )
    
    def prepare_prompt(self, 
                      content: str, 
                      context: list[Message] | None = None) -> str:
        """Prepare the prompt for the LLM including relevant context.
        
        Args:
            content: The current message to respond to
            context: Optional additional context messages
            
        Returns:
            Formatted prompt string
        """
        messages = []
        
        if self.system_prompt:
            messages.append(f"System: {self.system_prompt}")
            
        if context:
            # Add relevant context messages, possibly with some filtering/selection
            for msg in context:
                messages.append(
                    f"From {msg.from_model} to {msg.to_model}: {msg.content}"
                )
        
        messages.append(f"Current message: {content}")
        return "\n\n".join(messages)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text. To be implemented with actual tokenizer."""
        # TODO: Implement actual token counting
        return len(text.split())  # Naive approximation for now
    
    def forget_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history.clear()
