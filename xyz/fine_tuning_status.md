# Status Update Log

## v0.1.8 - Increased Training Iteration Speed, Adjusting for Duration

- Increased iteration speed from 10s/iteration to 15s/iteration.
- However, the number of iterations also increased from 5000 to 8000, resulting in an estimated training duration of 32 hours.
- Searching for new ways to decrease overall training time.
- Attached are screenshots of progress for two different versions (try1 and try2) for comparison.

## v0.1.7 - Prevented System Hibernation (Note Added)

- Added a note: Remember, these changes will persist across reboots. If you ever need to re-enable these power-saving features in the future, you can unmask them using:
  ```bash
  sudo systemctl unmask sleep.target suspend.target hibernate.target hybrid-sleep.target
  ```

## v0.1.6 - Prevented System Hibernation

- Executed the following command to prevent the system from entering sleep, suspend, or hibernation states:
  ```bash
  sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
  ```
- Created symlinks for `/etc/systemd/system/sleep.target`, `/suspend.target`, `/hibernate.target`, and `/hybrid-sleep.target` to `/dev/null`.
- This will ensure uninterrupted operations during long-running training tasks.

## v0.1.5 - Memory Allocation Challenges and Solutions

- Encountered "CUDA out of memory" errors during fine-tuning of LLaMA model.
- Diagnosis:
  - Used custom GPU memory check script to understand memory usage.
  - Found reported memory usage didn't match actual available memory on `g4dn.xlarge` instance (16 GB of GPU memory).
- Solutions Implemented:
  - **Quantization**: Used 8-bit quantization with `BitsAndBytesConfig` to reduce memory footprint.
  - **Gradient Checkpointing**: Enabled to trade computation for memory.
  - **Mixed Precision Training**: Enabled fp16 training to reduce memory usage.
  - **Parameter-Efficient Fine-Tuning (PEFT)**: Implemented LoRA using `peft` library to add small, trainable adapters.
  - **Optimizer Adjustments**: Used `paged_adamw_8bit` optimizer to offload optimizer states to CPU.
  - **Gradient Accumulation**: Implemented to simulate larger batch sizes without increasing memory usage.
- Updated model loading to include quantization and PEFT.
  - Example code:
    ```python
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(...)
    model = get_peft_model(model, peft_config)
    ```
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to help with memory fragmentation.
- Successfully began training the model on the `g4dn.xlarge` instance with 16 GB GPU memory.

## v0.1.4 - Fine-Tuning Preparation and Key Considerations

- Using `load_from_disk` to load the preprocessed SQuAD dataset.
- Created a custom `SQuADDataset` class to format the data for question answering. It constructs input-output pairs suitable for the LLaMA model.
- Input format: "Question: [question] Context: [context] Answer:"; Target: the answer text.
- Reduced batch size to 4 due to the large size of the LLaMA model, adjustable based on GPU memory.
- Learning rate and other hyperparameters may need tuning for optimal performance.
- Saving checkpoints during training and limiting total number of saved checkpoints to manage disk space.
- After training, the fine-tuned model and tokenizer will be saved.
- Key considerations before running the script:
  - Ensure sufficient GPU memory; fine-tuning LLaMA-3.2-1B requires significant resources.
  - Install additional dependencies: `pip install transformers datasets torch`.

## v0.1.3 - Verified Model Path and Loaded Model Successfully

- Created `verify_model_path.py` to check and load the LLaMA 3.2-1B model.
- Successfully verified the model directory contents, including `config.json`, `generation_config.json`, `model.safetensors`, and tokenizer files.
- Loaded the tokenizer and model without any issues, confirming the setup.

## v0.1.2 - Ensured Dataset Correctness

- The output confirmed that the SQuAD dataset is correctly formatted and contains the expected information.
- Dataset Structure:
  - Two splits: 'train' and 'validation'.
  - Training set: 87,599 examples; Validation set: 10,570 examples.
- Data Fields:
  - Fields include 'id', 'title', 'context', 'question', and 'answers'.
- Sample data looks appropriate for a question-answering task.
- Data is already in a preprocessed format suitable for fine-tuning the LLaMA model.
- Next Steps:
  - Examined sample data in `sample_data.json`.
  - Planned to prepare data for fine-tuning by loading with Hugging Face, tokenizing, creating a DataLoader, and setting up a training loop.
  - Set up evaluation process to test model performance on validation set.

## v0.1.1 - New Project: Fine-Tuning with SQuAD Dataset

- Started a new project to fine-tune LLaMA-3.2-1B model using the SQuAD dataset.
- Outlined the steps for fine-tuning, including data preparation, model setup, training loop, and evaluation.
- Loaded the SQuAD dataset (`train-v2.0.json` and `dev-v2.0.json`).
- Set up the LLaMA model and tokenizer for the fine-tuning process.
- Defined training arguments using Hugging Face's `Trainer` class for efficient fine-tuning.
- Planned to evaluate the model on the validation dataset to assess performance.

## v0.0.9 - Hit a Wall with Model Compatibility

- Discovered that the LLaMA-3.2-1B model is text-only and does not have image processing capabilities.
- Realized a different type of multimodal model is required for image-text tasks (e.g., CLIP, ViLT, BLIP).
- Encountered a roadblock with plans for the image-text project using the current model.
- Considering the next steps for either switching to a multimodal model or continuing with text-based features.

## v0.0.8 - Installed Accelerate Library and Modified Model Loading Script

- Encountered missing `accelerate` library error while attempting to load the model.
- Installed `accelerate` library for efficient model loading:
  ```bash
  pip install 'accelerate>=0.26.0'
  ```
- Modified `test_model_loading.py` script to remove `device_map="auto"` to bypass the need for `accelerate` in the interim.
- Tested the modified script for model loading and inference.

## v0.0.7 - Downgraded Model and Completed Download

- Attempted to run `run_inference.py`, but the model couldn't be initialized due to memory limitations.
- Discovered that the `g4dnxlarge` instance isn't sufficient (16 GiB of GPU memory) for the 11B-instruct model, which requires at least 45 GiB.
- Debugged the issue, learned the current server configuration limits, and realized further instance upgrade requests would likely be declined due to limited AWS history (only 1 day).
- Decided to pivot to a smaller model—1B-instruct.
- Faced challenges with downloading expired access to Meta’s 11B model after 48 hours.
- Obtained new access, mistakenly tried downloading from Hugging Face, and had to set up access tokens for a different download approach.
- Finally downloaded the LLaMA 1B model for continuing the project.

## v0.0.6 - Processed Image Data

- Ran `process_data.py` script successfully.
- Image data saved to `image_data.json`.

## v0.0.5 - Moving Image Dataset to Server

- Moved `image_dataset` to the AWS server to start working with images directly from the instance.

## v0.0.4 - JSON Metadata Created

- Generated JSON metadata for over 20 images to be used in the multi-modal project.

## v0.0.3 - Environment Set Up

- Reinstalled the virtual environment with essential dependencies:
  - `torch==2.4.1`
  - `torchvision==0.19.1`
  - `transformers==4.45.2`
  - `datasets==3.0.1`
  - `sentencepiece==0.2.0`
  - `llama-stack==0.0.42`

## v0.0.2 - Restarted AWS Instance and Connected via SSH

- Restarted the AWS EC2 instance and updated the public IP address for SSH and VS Code connection.
- Successfully connected using VS Code Remote SSH.

## v0.0.1 - AWS Setup Completed

- Set up an AWS EC2 instance (`g4dn.xlarge`), configured security groups, and installed all required libraries for running Meta’s LLaMA 3.2 Vision Instruct model.

