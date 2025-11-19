"""
Configuration Module for AI Model Application
==============================================
This module defines settings and configurations for an AI model system.
It uses dataclasses and enums to create type-safe, organized configurations.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# MODEL PROVIDER ENUM

# Purpose: Define which AI service/platform will run the models
# 
# Visual Structure:
#   ModelProvider (Enum)
#   └── OLLAMA = "ollama"  ← Currently only supports Ollama (local AI runner)
#
# Why use Enum?
# - Type safety: Only valid providers can be used
# - Easy to extend: Just add more providers like OPENAI = "openai"
# - Prevents typos: Can't accidentally write "olama" instead of "ollama"

class ModelProvider(str, Enum):
    """
    Enumeration of supported AI model providers.
    
    Inherits from both str and Enum to allow string comparisons while
    maintaining the benefits of enumeration (type safety, autocomplete).
    """
    OLLAMA = "ollama"  # Local AI model runner (https://ollama.ai)
    # Future providers could be added here:
    # OPENAI = "openai"
    # ANTHROPIC = "anthropic"

# MODEL CONFIGURATION DATACLASS

# Purpose: Bundle all settings for a specific AI model into one object
#
# Visual Structure:
#   ModelConfig
#   ├── name: str          ← Model identifier (e.g., "qwen3:8b")
#   ├── temperature: float ← Randomness level (0.0 = deterministic, 1.0 = creative)
#   └── provider: Enum     ← Which service runs this model
#
# Why use @dataclass?
# - Auto-generates __init__, __repr__, __eq__ methods
# - Clean syntax for creating configuration objects
# - Type hints for better IDE support and error checking

@dataclass
class ModelConfig:
    """
    Configuration for an AI language model.
    
    Attributes:
        name: The model identifier (e.g., "qwen3:8b" for Qwen 3 with 8B parameters)
        temperature: Controls randomness in responses
                     • 0.0 = Deterministic (same input → same output)
                     • 0.5 = Balanced creativity
                     • 1.0 = Maximum creativity/randomness
        provider: Which service/platform hosts this model
    """
    name: str              # Model identifier string
    temperature: float     # Range: 0.0 (deterministic) to 1.0 (creative)
    provider: ModelProvider  # Must be a valid ModelProvider enum value

# PREDEFINED MODEL CONFIGURATIONS

# Purpose: Ready-to-use model configurations
#
# QWEN_3 Configuration:
#   Model: qwen3:8b (Qwen 3 with 8 billion parameters)
#   Temperature: 0.0 (deterministic - perfect for testing/reproducibility)
#   Provider: Ollama (running locally)

QWEN_3 = ModelConfig("qwen3:8b", temperature=0.0, provider=ModelProvider.OLLAMA)
# This creates an instance you can use directly without recreating it

# MAIN APPLICATION CONFIGURATION

# Purpose: Central hub for all application-wide settings
#
# Visual Architecture:
#   Config
#   ├── SEED = 42              ← Random seed for reproducibility
#   ├── MAX_ITERATIONS = 10    ← Loop/retry limit
#   ├── MODEL = QWEN_3         ← Which AI model to use
#   ├── CONTEXT_WINDOW = 8192  ← Max tokens the model can process at once
#   └── Path (nested class)
#       ├── APP_HOME    ← Root directory of the application
#       └── DATA_DIR    ← Where data files are stored
#
# Why use a class for config instead of a dict?
# - Namespacing: Config.SEED is clear and organized
# - Type safety: Can add type hints
# - IDE support: Better autocomplete and refactoring
# - No accidental overwrites: Class constants are more protected

class Config:
    """
    Master configuration class for the application.
    
    Contains all global settings and paths used throughout the application.
    Using a class (instead of individual variables) keeps everything organized
    and prevents naming conflicts.
    """
    
    # REPRODUCIBILITY SETTINGS
    SEED = 42  # Random seed for reproducible results
               # Same seed = same random sequences = reproducible experiments
    
    # EXECUTION LIMITS
    MAX_ITERATIONS = 10  # Maximum number of loops/retries/generations
                         # Prevents infinite loops and controls costs
    
    # AI MODEL SETTINGS
    MODEL = QWEN_3  # Active AI model configuration
                    # Change this to switch models throughout the app
    
    CONTEXT_WINDOW = 8192  # Maximum tokens (words/pieces) the model can handle
                           # Token ≈ 0.75 words on average
                           # 8192 tokens ≈ 6,144 words ≈ 12 pages of text
                           # Exceeding this causes errors or truncation
    
    # FILE SYSTEM PATHS (Nested Class)
    # Nested classes help organize related settings into logical groups
    class Path:
        """
        File system path configurations.
        
        Uses pathlib.Path for cross-platform compatibility (works on
        Windows, Mac, Linux without modification).
        """
        
        # Root directory of the application
        # Flow: 
        # 1. Try to read APP_HOME from environment variable
        # 2. If not set, use this file's parent directory's parent
        #    (goes up two levels: config.py -> src/ -> app_root/)
        APP_HOME = Path(os.getenv("APP_HOME", Path(__file__).parent.parent))
        #                         ↑                    ↑
        #                         |                    |
        #                  Environment Variable   Default Fallback
        #
        # Why environment variables?
        # - Flexibility: Different paths for dev/production/testing
        # - Security: Keep sensitive paths out of code
        # - Docker/Cloud: Easy to configure in deployment
        
        # Data directory where files are stored
        DATA_DIR = APP_HOME / "data"  # The / operator joins paths safely
                                       # Equivalent to: APP_HOME.joinpath("data")
                                       # Example: /home/user/app/data