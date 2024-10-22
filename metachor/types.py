from dataclasses import dataclass
from enum import Enum

class Phase(Enum):
    """Phases of the collaborative response generation process."""
    INITIALIZATION = "initialization"
    USER_ANALYSIS = "user_analysis"
    RESPONSE_PLANNING = "planning"
    RESPONSE_DRAFTING = "drafting"
    RESPONSE_REFINING = "refining"

# Define phase contexts at the type level since they're intrinsic to the phases
PHASE_CONTEXTS = {
    Phase.INITIALIZATION: "Briefly acknowledge other models and confirm readiness",
    Phase.USER_ANALYSIS: "What are the key aspects of this request we need to address?",
    Phase.RESPONSE_PLANNING: "What's the most effective way to structure our response?",
    Phase.RESPONSE_DRAFTING: "Generate the response following our plan",
    Phase.RESPONSE_REFINING: "Review and improve the drafted response"
}

@dataclass
class ResourceConstraints:
    """Constraints for a collaborative response generation session."""
    max_tokens: int
    max_iterations: int
    max_time: float  # seconds

@dataclass
class Message:
    """A message exchanged between models in the ensemble."""
    content: str
    tokens_used: int
    from_model: str
    to_model: str
    phase: Phase