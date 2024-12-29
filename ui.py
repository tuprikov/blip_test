"""
This module provides a simple interface to launch a Gradio interface for a given function.
"""
from typing import Callable

import gradio as gr


def launch(func: Callable, inputs: list, outputs: str):
    """Launch a Gradio interface for a given function."""
    iface = gr.Interface(
        fn=func,
        inputs=inputs,
        outputs=outputs,
        title="Image Captioning with BLIP",
        description="Upload an image to generate a caption."
    )
    iface.launch()
