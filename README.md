Custom-medgemma-REG2025

This repository manages custom fine-tuning and evaluation tasks for pathology image analysis and automated report generation using the MedGemma model. MedGemma combines pathological images and medical text to generate clinically valid, automated pathology reports.

Repository Structure

.ipynb_checkpoints: Directory for Jupyter Notebook checkpoint files.

fintuning: Directory containing custom fine-tuning code for the MedGemma model.

fintuning_result: Directory containing code that generates JSON outputs from the fine-tuned model.

test: Directory for test data and model evaluation.

Quick Start

The following usage examples are provided:

Quick start with Hugging Face: Example of generating responses from medical text and images locally using Hugging Face.

Quick start with Model Garden: Example of serving the model on Vertex AI for generating responses from medical text and images in online or batch workflows.

Fine-tune with Hugging Face: Example of fine-tuning the model with LoRA using Hugging Face libraries.

Additional Notice

Due to large file sizes, data and models have not been stored directly in this repository.



# MedGemma Notebooks (quick start)

*   [Quick start with Hugging Face](quick_start_with_hugging_face.ipynb) -
    Example of generating responses from medical text and images by running the
    model locally from Hugging Face.

*   [Quick start with Model Garden](quick_start_with_model_garden.ipynb) -
    Example of serving the model on
    [Vertex AI](https://cloud.google.com/vertex-ai/docs/predictions/overview)
    and using Vertex AI APIs to generate responses from medical text and images
    in online or batch workflows.

*   [Fine-tune with Hugging Face](fine_tune_with_hugging_face.ipynb) - Example
    of fine-tuning the model with LoRA using Hugging Face libraries.
