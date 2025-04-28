"""
Sub-Class for 
Model: Trellis by Microsoft
Target: Image to 3D

Created on: 16-04-2025

Last Modified: 16-04-2025
"""

import os
from pathlib import Path
from typing import *
from Trellis_ModelGeneration import Trellis_ModelGeneration, ModelInit, RunConfig, ExportGLBConfig

import imageio
from rembg import remove
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


class Trellis_i23D_ModelGeneration(Trellis_ModelGeneration):
    """
    This subclass handles the generation of 3D models from images using the Trellis model.
    It extends the `Trellis_ModelGeneration` class to specialize in image-to-3D transformation.
    
    Args:
    model (ModelInit): Configuration object for model initialization containing the model type, size, and output path.
    """

    def __init__(self, model: ModelInit):
        """
        Initializes the Trellis image-to-3D model generation pipeline.

        Args:
        model (ModelInit): Configuration object for initializing the model with necessary parameters.
        """
        # Set model type to image
        model.model_type = 'image'  
        super().__init__(model)

        # Load the pretrained pipeline for image-to-3D conversion
        self.pipeline = self._load_pipeline()

        # Variables to store results and metadata
        self.outputs = None
        self.pipeline_attributes: Dict[str, Any] = {}
        self.global_attributes: Dict[str, Any] = {}
        self.image_path = None

        # Default parameters for sparse structure and SLAT samplers
        self.config_structure_sampler_default = {
            "steps": 12,
            "cfg_strength": 7.5
        }
        self.config_slat_sampler_default = {
            "steps": 12,
            "cfg_strength": 3
        }

    def set_args_RunConfig(self, config: RunConfig) -> Dict[str, Any]:
        """
        Extracts and prepares the pipeline arguments from the provided `RunConfig` object.

        Args:
        config (RunConfig): Configuration object that contains parameters like seed, number of samples, and more.

        Returns:
        dict: A dictionary containing the prepared arguments for running the pipeline.
        """
        args = {}
        args['sparse_structure_sampler_params'] = config.sparse_structure_sampler_params or self.config_structure_sampler_default
        args['slat_sampler_params'] = config.slat_sampler_params or self.config_slat_sampler_default
        if config.mode is not None:
            args['mode'] = config.mode
        if config.seed is not None:
            args['seed'] = config.seed
        if config.num_samples is not None:
            args['num_samples'] = config.num_samples
        return args

    def _load_pipeline(self):
        """
        Loads the pretrained Trellis pipeline for image-to-3D conversion and prepares it for use.

        Args:
        None

        Returns:
        pipeline (TrellisImageTo3DPipeline): The loaded Trellis image-to-3D pipeline.
        """
        pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_path)
        pipeline.cuda()  # Moves the pipeline to GPU for faster processing
        return pipeline

    def load_image(self, image_path: str) -> Image.Image:
        """
        Loads an image from the provided file path and returns it as a PIL Image object.
        Raises a FileNotFoundError if the provided path does not exist.

        Args:
        image_path (str): The file path of the image to be loaded.

        Returns:
        Image.Image: A PIL Image object representing the loaded image.
        
        Raises:
        FileNotFoundError: If the image file does not exist at the given path.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        self.image_path = image_path
        return Image.open(image_path)

    def run(self, image: Image.Image, config: RunConfig) -> dict:
        """
        Runs the image-to-3D conversion pipeline on a single image using the provided configuration.

        Args:
        image (Image.Image): The image to be processed into a 3D model.
        config (RunConfig): Configuration object containing the necessary parameters for running the pipeline.

        Returns:
        dict: The generated 3D model output after processing the image.
        """
        args = self.set_args_RunConfig(config)
        self.outputs = self.pipeline.run(image, **args)
        return self.outputs

    def run_multi(self, image: List[Image.Image], config: RunConfig) -> dict:
        """
        Runs the image-to-3D conversion pipeline on multiple images.

        Args:
        image (List[Image.Image]): A list of images to be processed into 3D models.
        config (RunConfig): Configuration object containing the necessary parameters for running the pipeline.

        Returns:
        dict: The generated 3D models output after processing the list of images.
        """
        args = self.set_args_RunConfig(config)
        self.outputs = self.pipeline.run_multi_image(image, **args)
        return self.outputs

    def export_glb(self, config: ExportGLBConfig) -> str:
        """
        Exports the generated 3D model output to a GLB file, using the specified export configuration.

        Args:
        config (ExportGLBConfig): The configuration specifying how the GLB file should be exported (e.g., file name, texture size, etc.).

        Returns:
        str: The file path to the exported GLB file.
        """
        # Convert pipeline outputs to GLB format using post-processing utilities
        glb = postprocessing_utils.to_glb(
            self.outputs["gaussian"][0],
            self.outputs["mesh"][0],
            simplify=config.simplify,
            texture_size=config.texture_size,
            mode=config.mode
        )

        # Determine the output path for the GLB file
        if config.output_path is None:
            output_path = Path(self.base_output_path) / f"{Path(self.image_path).stem}.glb"
        else:
            output_path = Path(config.output_path)
            if output_path.suffix == "":
                # If only a directory is provided, append the filename to form a full path
                output_path /= f"{Path(self.image_path).stem}.glb"
            output_path = Path(self.base_output_path) / output_path

        # Create the necessary directories for the output file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the GLB file to the determined path
        glb.export(str(output_path))
        return str(output_path)

    def set_pipeline_attribute(self, key: str, value: Any):
        """
        Sets a custom attribute for the pipeline, which can be used to influence the pipeline's behavior.

        Args:
        key (str): The attribute name.
        value (Any): The value to assign to the attribute.
        """
        self.pipeline_attributes[key] = value

    def get_pipeline_attribute(self, key: str) -> Any:
        """
        Retrieves a custom attribute from the pipeline.

        Args:
        key (str): The name of the attribute to retrieve.

        Returns:
        Any: The value of the requested attribute, or None if the attribute does not exist.
        """
        return self.pipeline_attributes.get(key)

    def set_global_attribute(self, key: str, value: Any):
        """
        Sets a global attribute for general use throughout the model generation process.

        Args:
        key (str): The global attribute name.
        value (Any): The value to assign to the global attribute.
        """
        self.global_attributes[key] = value

    def get_global_attribute(self, key: str) -> Any:
        """
        Retrieves a global attribute.

        Args:
        key (str): The name of the global attribute to retrieve.

        Returns:
        Any: The value of the requested global attribute, or None if the attribute does not exist.
        """
        return self.global_attributes.get(key)
