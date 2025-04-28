"""
Sub-Class for 
Model: Trellis by Microsoft
Target: Image to 3D

Created on: 16-04-2025

Last Modified: 16-04-2025
"""
import os
from pathlib import Path
from typing import Any, Dict
from Trellis_ModelGeneration import Trellis_ModelGeneration, ModelInit, RunConfig, ExportGLBConfig

import imageio
from rembg import remove
from PIL import Image
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils


class Trellis_i23D_ModelGeneration(Trellis_ModelGeneration):
    def __init__(self, model: ModelInit):
        model.model_type = 'image'
        super().__init__(model)
        self.pipeline = self._load_pipeline()
        self.outputs = None
        self.pipeline_attributes: Dict[str, Any] = {}
        self.global_attributes: Dict[str, Any] = {}

    def _load_pipeline(self):
        pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_path)
        pipeline.cuda()
        return pipeline

    def run(self, prompt: str, config: RunConfig):

        if config.sparse_structure_sampler_params is None:
            config.sparse_structure_sampler_params = {
                "steps": 12,
                "cfg_strength": 7.5
                }
        if config.slat_sampler_params is None:
            config.slat_sampler_params = {
                "steps": 12,
                "cfg_strength": 3
                }

        self.outputs = self.pipeline.run(
            prompt,
            seed=config.seed,
            sparse_structure_sampler_params=config.sparse_structure_sampler_params,
            slat_sampler_params=config.slat_sampler_params,
        )
        return self.outputs


    def export_glb(self, config: ExportGLBConfig) -> str:
        glb = postprocessing_utils.to_glb(
            self.outputs["gaussian"][0],
            self.outputs["mesh"][0],
            simplify=config.simplify,
            texture_size=config.texture_size,
        )
        if config.output_path is None:
            config.output_path = Path(f'{self.base_output_path}/{Path(self.image_path).stem}.glb')
        glb.export(config.output_path)
        return config.output_path

    def set_pipeline_attribute(self, key: str, value: Any):
        self.pipeline_attributes[key] = value

    def get_pipeline_attribute(self, key: str) -> Any:
        return self.pipeline_attributes.get(key)

    def set_global_attribute(self, key: str, value: Any):
        self.global_attributes[key] = value

    def get_global_attribute(self, key: str) -> Any:
        return self.global_attributes.get(key)

   