"""
This module provides a simple interface to launch a Gradio interface for a given function.
"""
from typing import Callable

import gradio as gr


def launch(func: Callable, inputs: list, outputs):
    """Launch a Gradio interface for a given function."""
    iface = gr.Interface(
        fn=func,
        inputs=inputs,
        outputs=outputs,
    )
    iface.launch()
