# metachor/types.py
from dataclasses import dataclass
from enum import Enum

class Phase(Enum):
    """Phases of the collaborative response generation process."""
    INITIALIZATION = "initialization"  # Initial meta-discussion about nature/roles
    USER_ANALYSIS = "user_analysis"    # Analyzing user request and constraints
    RESPONSE_PLANNING = "planning"     # Planning optimal response approach
    RESPONSE_DRAFTING = "drafting"     # Iteratively developing response
    RESPONSE_REFINING = "refining"     # Final refinements and optimization

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
