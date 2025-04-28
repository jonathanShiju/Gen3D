"""
Class Definition for 
Model: Trellis by Microsoft

Author: Jonathan Shiju
Created on: 16-04-2025

Last Modified: 16-04-2025
"""

'''
Pydantic for Data Validation
'''
from pathlib import Path
from typing import *
from pydantic import BaseModel, Field, validator, model_validator
import os

# Model initialization schema using Pydantic for validation
class ModelInit(BaseModel):
    """
    Model initialization configuration for Trellis model.

    Args:
    model_type (str): The type of model to use. Valid options are 'image' or 'text'. Default is 'image'.
    model_size (str): The size of the model. Valid options are 'base', 'large', or 'xlarge'. Default is 'large'.
    base_output_path (str, optional): Custom base output path where generated models are stored. Defaults to None.

    Expected output:
    ModelInit object with validated model parameters.
    """
    model_type : Literal["image", "text"] = "image"  # Type of model to use (image or text)
    model_size : Literal["base", "large", "xlarge"] = "large"  # Size of the model
    base_output_path : Optional[str] = None  # Optional custom output path

    @model_validator(mode="after")
    def model_size_valid(self) -> "ModelInit":
        """
        Custom validator to ensure only supported image model sizes are used.
        If the model type is 'image' and the model size is unsupported, it defaults to 'large'.

        Args:
        None (validated at the time of initialization)

        Returns:
        ModelInit: Returns the ModelInit object with adjusted model size if necessary.
        """
        if self.model_type == "image":
            if self.model_size in ["xlarge", "base"]:
                self.model_size = "large"  # Default to 'large' if unsupported size is chosen
                print("Model size not supported for image model. Defaulting to large.")
        return self

# Configuration for running the model generation process
class RunConfig(BaseModel):
    """
    Configuration for running the model generation process.

    Args:
    seed (int): Random seed for reproducibility. Default is 1.
    sparse_structure_sampler_params (dict, optional): Parameters for the sparse structure sampler. Defaults to None.
    slat_sampler_params (dict, optional): Parameters for the SLAT sampler. Defaults to None.
    num_samples (int): Number of samples to generate. Default is 1.
    mode (str, optional): Generation mode; can be 'stochastic' or 'multidiffusion'. Defaults to None.

    Expected output:
    RunConfig object with the configuration for model generation.
    """
    seed: int = 1  # Random seed for reproducibility
    sparse_structure_sampler_params: Optional[Dict[str, Any]] = None  # Params for sparse structure sampler
    slat_sampler_params: Optional[Dict[str, Any]] = None  # Params for SLAT sampler
    num_samples: int = 1  # Number of samples to generate
    mode: Optional[Literal['stochastic', 'multidiffusion']] = None  # Generation mode

# Configuration for exporting generated 3D models to GLB format
class ExportGLBConfig(BaseModel):
    """
    Configuration for exporting generated models to GLB format.

    Args:
    output_path (str, optional): Path to save the exported GLB file. Defaults to None.
    simplify (float): Simplification factor for the 3D mesh, with a default of 0.95.
    texture_size (int): Size of the texture map (default is 1024).
    mode (str): Export mode; can be 'opt' (optimized) or 'fast'. Defaults to 'opt'.

    Expected output:
    ExportGLBConfig object with the configuration for GLB export.
    """
    output_path: Optional[Path] = None  # Path to save the exported GLB file
    simplify: float = 0.95  # Simplification factor for the 3D mesh
    texture_size: int = 1024  # Size of the texture map
    mode: Literal['opt', 'fast'] = 'opt'  # Export mode: optimized or fast

# Core class for handling Trellis model generation
class Trellis_ModelGeneration:
    """
    Main class responsible for initializing and generating 3D models using the Trellis model.

    Args:
    model (ModelInit): ModelInit object containing the configuration for the Trellis model (model type, size, output path).

    Expected output:
    Trellis_ModelGeneration object that can be used to generate and export 3D models.
    """
    def __init__(self, model: ModelInit):
        """
        Initializes the model generation class. Sets up the necessary environment variables and directories for output.
        Args:
        model (ModelInit): The configuration object containing the model parameters.
        """
        # Set sparse convolution algorithm to native to optimize model generation
        os.environ['SPCONV_ALGO'] = 'native'

        # Construct model path based on model type and size (e.g., 'TRELLIS-image-large')
        self.model_path = f'TRELLIS-{model.model_type}-{model.model_size}'

        # Determine base output path (either user-defined or default to a temporary workspace)
        if model.base_output_path is None:
            # Default output path if none is provided
            path = Path(f'/workspace/temp/{model.model_type}')
            path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            self.base_output_path = path
        else:
            # Use user-provided output path
            path = Path(model.base_output_path)
            path.mkdir(parents=True, exist_ok=True)
            self.base_output_path = path

        # The object now has the initialized model path and output path, ready for model generation.
