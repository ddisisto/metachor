# metachor/cli.py

import os
import time
import asyncio
import httpx
import logging
from pathlib import Path
from typing import Annotated, Optional
import typer
from rich.console import Console
from rich.logging import RichHandler
from dotenv import load_dotenv

from metachor.types import ResourceConstraints
from metachor.voice import Voice
from metachor.ensemble import Ensemble

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Create console for user-facing output
console = Console()

# Initialize loggers and handlers at module level
logging.getLogger().setLevel(logging.WARN)  # Set root logger to WARN by default
metachor_logger = logging.getLogger("metachor")
metachor_logger.setLevel(logging.INFO)  # Set metachor logger to INFO by default

file_handler = logging.FileHandler(
    logs_dir / f"metachor_{time.strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))

console_handler = RichHandler(
    rich_tracebacks=True,
    show_time=False,
    show_path=False
)
console_handler.setLevel(logging.WARN)

logging.getLogger().addHandler(console_handler)
metachor_logger.addHandler(file_handler)

# Get logger for this module
log = logging.getLogger("metachor")

app = typer.Typer(help="metachor - Cognition in concert")

# Load environment variables
load_dotenv()

def configure_logging(verbose: bool) -> None:
    """Configure logging levels based on verbosity."""
    if verbose:
        # Verbose mode
        logging.getLogger().setLevel(logging.INFO)
        metachor_logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.INFO)
        file_handler.setLevel(logging.DEBUG)
    else:
        # Default mode
        logging.getLogger().setLevel(logging.WARN)
        metachor_logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.WARN)
        file_handler.setLevel(logging.INFO)

@app.callback()
def app_callback(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False
) -> None:
    """Initialize app-wide settings and configure logging."""
    configure_logging(verbose)

@app.command()
def chat(
    prompt: Annotated[str, typer.Argument(help="The prompt to send to the ensemble")],
    models: Annotated[list[str], typer.Option("--model", "-m")] = ["mistralai/ministral-8b", "liquid/lfm-40b"],
    max_tokens: Annotated[int, typer.Option("--max-tokens", "-t")] = 1000,
    max_time: Annotated[float, typer.Option("--max-time")] = 30.0,
    skip_init: Annotated[bool, typer.Option("--skip-init")] = False,
):
    """Send a prompt to the ensemble and get a collaborative response."""
    log.info(f"Starting collaborative chat with models {models} - max_tokens: {max_tokens}, max_time: {max_time}s")
    try:
        ensemble = create_ensemble(models)
        asyncio.run(run_chat(ensemble, prompt, max_tokens, max_time))
    except Exception as e:
        log.error(f"Failed to initialize ensemble: {str(e)}", exc_info=True)
        raise typer.Exit(code=1)

@app.command()
def direct(
    prompt: Annotated[str, typer.Argument(help="The prompt to send directly to each model")],
    models: Annotated[list[str], typer.Option("--model", "-m")] = ["mistralai/ministral-8b", "liquid/lfm-40b"],
    max_tokens: Annotated[int, typer.Option("--max-tokens", "-t")] = 1000,
    max_time: Annotated[float, typer.Option("--max-time")] = 30.0,
):
    """Send a prompt directly to each model without collaboration."""
    log.info(f"Starting direct model queries with models {models} - max_tokens: {max_tokens}, max_time: {max_time}s")
    try:
        ensemble = create_ensemble(models)
        asyncio.run(run_direct(ensemble, prompt, max_tokens, max_time))
    except Exception as e:
        log.error(f"Failed to initialize direct mode: {str(e)}", exc_info=True)
        raise typer.Exit(code=1)

def create_ensemble(models: list[str]) -> Ensemble:  # Remove system_prompt parameter
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
            max_tokens=1000  # Maybe make this configurable
        )
        voices.append(voice)
    
    return Ensemble(voices)

async def run_chat(
    ensemble: Ensemble,
    prompt: str,
    max_tokens: int,
    max_time: float
) -> None:
    """Run a chat interaction and display the result."""
    try:
        # Create a status context that won't interfere with logging
        with console.status("[bold blue]Processing...", spinner="dots"):
            constraints = ResourceConstraints(
                max_tokens=max_tokens,
                max_iterations=10,
                max_time=max_time
            )
            
            log.debug(f"Processing request with constraints: {constraints}")
            start_time = time.time()
            
            try:
                response = await ensemble.send(
                    prompt, 
                    constraints,
                    include_initialization=True
                )
                elapsed = time.time() - start_time
                
                # Clear any status and display results
                console.print()  # Add blank line
                console.print("[bold green]Response:[/bold green]")
                console.print(response)
                console.print(f"\n[dim]Completed in {elapsed:.2f} seconds[/dim]")
                
            except asyncio.TimeoutError:
                log.warning("Response timed out")
                console.print("\n[yellow]Response timed out, but partial results may be available[/yellow]")
                partial_response = ensemble._format_final_response(ensemble._get_all_responses())
                if partial_response:
                    console.print(partial_response)
                    
            except asyncio.CancelledError:
                log.warning("Operation was cancelled")
                console.print("\n[yellow]Operation was cancelled, but partial results may be available[/yellow]")
                partial_response = ensemble._format_final_response(ensemble._get_all_responses())
                if partial_response:
                    console.print(partial_response)
                    
    except Exception as e:
        log.exception("Failed to complete chat")
        raise typer.Exit(code=1)

async def run_direct(
    ensemble: Ensemble,
    prompt: str,
    max_tokens: int,
    max_time: float
) -> None:
    """Run direct interactions with each model and display results."""
    try:
        with console.status("Processing direct responses...", spinner="dots"):
            constraints = ResourceConstraints(
                max_tokens=max_tokens,
                max_iterations=1,  # Direct mode only needs one iteration
                max_time=max_time
            )
            
            start_time = time.time()
            response = await ensemble.send_direct(prompt, constraints)
            elapsed = time.time() - start_time
            
            # Display results
            console.print("\n[bold green]Direct Responses:[/bold green]")
            console.print(response)
            console.print(f"\n[dim]Completed in {elapsed:.2f} seconds[/dim]")
            
    except Exception as e:
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.exception("Failed to complete direct mode:")
        else:
            log.error(f"Failed to complete direct mode: {str(e)}")
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