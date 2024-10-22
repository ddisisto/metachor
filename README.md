# metachor
*Cognition in concert* ∿⟷∿→✧

A collaborative LLM system where multiple models engage in structured dialogue to produce more comprehensive and thoughtful responses.

## Core Concept
metachor orchestrates multiple language models in a collaborative dialogue, allowing them to:
1. Discuss and analyze the user's request
2. Plan a comprehensive response strategy
3. Iteratively develop and refine the response
4. Maintain resource awareness throughout the process

## Quick Start
```bash
# Install
git clone https://github.com/yourusername/metachor.git
cd metachor
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
pip install -r requirements.txt

# Configure
# Create .env file with your OpenRouter API key:
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Run
python -m metachor.cli chat "Explain the concept of emergence"
```

## Usage Examples

Basic chat:
```bash
python -m metachor.cli chat "Explain quantum computing"
```

Specify models to use:
```bash
python -m metachor.cli chat "Analyze this poem" \
    -m anthropic/claude-3-opus \
    -m openai/gpt-4-turbo-preview
```

List available models and their costs:
```bash
python -m metachor.cli list-models
```

Adjust response constraints:
```bash
python -m metachor.cli chat "Write a short story" \
    --max-tokens 2000 \
    --max-time 60
```

Enable verbose logging:
```bash
python -m metachor.cli chat "Explain neural networks" -v
```

## Project Structure
```
metachor/
├── metachor/
│   ├── __init__.py
│   ├── types.py        # Core type definitions
│   ├── voice.py        # Individual LLM interface
│   ├── ensemble.py     # Orchestration logic
│   └── cli.py         # Command-line interface
└── tests/
    └── test_metachor.py
```

## Current Features
- Asynchronous API communication
- Support for multiple LLM providers via OpenRouter
- Resource-aware response generation
- Structured collaboration phases
- Rich command-line interface
- Comprehensive error handling and logging

## Development Status
metachor is in active development. Current focus areas:
- Refined collaboration strategies
- Enhanced response integration
- Local model support
- Conversation persistence
- Performance optimization

## Technical Notes
- Requires Python 3.12+
- Uses asyncio for concurrent operations
- Implements robust error handling
- Resource constraints are strictly enforced
- All API interactions are logged when verbose mode is enabled

## Implementation Details
The system operates in distinct phases:
1. **Initialization**: Models establish common ground and discuss their roles
2. **Analysis**: Collaborative examination of the user's request
3. **Planning**: Strategic development of response approach
4. **Generation**: Iterative response development with continuous refinement
5. **Integration**: Synthesis of multiple model contributions

Resource management is handled through:
- Token counting and limits
- Maximum iteration constraints
- Time-based boundaries
- Adaptive phase transitions

## Future Directions
- Support for local model deployment
- Enhanced collaboration patterns
- Persistent conversation context
- Custom collaboration strategies
- Performance optimizations
- Extended model support