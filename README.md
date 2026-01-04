# Fine_Tuning
Finetuning an LLM with specific data and application (To be decided)
# Integrating relavent API into the model

## Setup

1. Ensure you have Python installed.
2. Configure the Python environment (already done).
3. Install dependencies: `pip install -r requirements.txt` (already done).

## Hugging Face Setup

1. Create an account on [Hugging Face](https://huggingface.co).
2. Accept the license for [Gemma 3 1B IT](https://huggingface.co/google/gemma-3-1b-it).
3. Generate an access token from your [settings](https://huggingface.co/settings/tokens).
4. In `fine_tune.py`, replace `"your_hf_token"` with your actual token.

## Running Fine-Tuning

Run the script: `python fine_tune.py`

This will fine-tune the Gemma 3 1B model using QLoRA (set `use_qlora = False` for LoRA).

## Notes

- The script uses a small demo dataset. Replace `dataset_name` with your own dataset.
- Ensure you have sufficient GPU memory (at least 8GB for QLoRA).
- Training may take time depending on your hardware.
