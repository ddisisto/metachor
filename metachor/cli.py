# metachor/cli.py
import os
import time
import asyncio
import httpx
import logging
from typing import Annotated
import typer
from rich.console import Console
from rich.logging import RichHandler
from dotenv import load_dotenv

from metachor.types import ResourceConstraints
from metachor.voice import Voice
from metachor.ensemble import Ensemble

# Set up rich logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("metachor")
console = Console()

app = typer.Typer(help="metachor - Cognition in concert")

# Load environment variables
load_dotenv()

def create_ensemble(models: list[str], system_prompt: str | None = None) -> Ensemble:
    """Create an ensemble from a list of model IDs."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise typer.BadParameter(
            "OPENROUTER_API_KEY not found in environment variables. "
            "Please set it in your .env file."
        )
    
    voices = []
    for model in models:
        voice = Voice(
            model_id=model,
            api_key=api_key,
            system_prompt=system_prompt,
            max_tokens=1000  # Maybe make this configurable
        )
        voices.append(voice)
    
    return Ensemble(voices, system_prompt)

async def run_chat(
    ensemble: Ensemble,
    prompt: str,
    max_tokens: int,
    max_time: float
) -> None:
    """Run a chat interaction and display the result."""
    try:
        with console.status("Processing...", spinner="dots"):
            constraints = ResourceConstraints(
                max_tokens=max_tokens,
                max_iterations=10,
                max_time=max_time
            )
            
            # Initialize the ensemble's meta-discussion
            log.info("Initializing meta-discussion...")
            await ensemble.initialize_meta_discussion()
            
            # Send the actual prompt
            log.info("Sending prompt...")
            start_time = time.time()
            response = await ensemble.send(prompt, constraints)
            elapsed = time.time() - start_time
            
            # Display results
            console.print("\n[bold green]Response:[/bold green]")
            console.print(response)
            console.print(f"\n[dim]Completed in {elapsed:.2f} seconds[/dim]")
            
    except Exception as e:
        log.error(f"Error during chat: {str(e)}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def chat(
    prompt: Annotated[str, typer.Argument(help="The prompt to send to the ensemble")],
    models: Annotated[list[str], typer.Option(
        "--model", "-m", 
        help="Model identifiers (can be specified multiple times)"
    )] = ["mistralai/ministral-8b", "liquid/lfm-40b"],            # Cheap / diverse ones selected by daniel
    # ["anthropic/claude-3-opus", "openai/gpt-4-turbo-preview"],  # Defaults suggested by claude
    max_tokens: Annotated[int, typer.Option(
        "--max-tokens", "-t",
        help="Maximum tokens in response"
    )] = 1000,
    max_time: Annotated[float, typer.Option(
        "--max-time", 
        help="Maximum time in seconds"
    )] = 30.0,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose logging"
    )] = False,
):
    """
    Send a prompt to the ensemble and get a collaborative response.
    """
    if verbose:
        log.setLevel(logging.DEBUG)
    
    # Basic system prompt - we could make this configurable
    system_prompt = """You are part of an ensemble of AI models working together to provide
    comprehensive and thoughtful responses. Collaborate with your peers, building on each
    other's insights while maintaining a coherent narrative."""
    
    try:
        ensemble = create_ensemble(models, system_prompt)
        asyncio.run(run_chat(ensemble, prompt, max_tokens, max_time))
    except Exception as e:
        log.error(f"Failed to initialize ensemble: {str(e)}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def list_models():
    """List available models from OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY not found in environment variables.[/red]")
        raise typer.Exit(code=1)
    
    async def fetch_models():
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            log.debug(f"API Response: {data}")  # Debug log to see the response structure
            return data.get("data", [])  # OpenRouter wraps models in a 'data' field
    
    try:
        models = asyncio.run(fetch_models())
        console.print("\n[bold]Available Models:[/bold]\n")
        for model in models:
            model_id = model.get("id", "Unknown")
            context_length = model.get("context_length", "Unknown")
            pricing = model.get("pricing", {})
            prompt_price = pricing.get("prompt", "Unknown")
            completion_price = pricing.get("completion", "Unknown")
            
            console.print(f"• {model_id}")
            console.print(f"  Context: {context_length} tokens")
            console.print(f"  Cost: ${prompt_price} per prompt token")
            console.print(f"        ${completion_price} per completion token\n")
    except Exception as e:
        log.error(f"Failed to fetch models: {str(e)}", exc_info=True)
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()