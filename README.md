# Dialogue Summarization with Fine-Tuned `google/flan-t5-base`

This repository demonstrates how to fine-tune the `google/flan-t5-base` model for dialogue summarization. The project's primary goal is to evaluate the effectiveness of different fine-tuning techniques, including full fine-tuning and parameter-efficient fine-tuning (PEFT), and compare their performance on the [dialogsum dataset](https://huggingface.co/datasets/knkarthick/dialogsum).

## Table of Contents
1. [Background](#background)
2. [Project Objectives](#project-objectives)
3. [Dataset](#dataset)
4. [Data Preparation](#data-preparation)
5. [Fine-Tuning Approaches](#fine-tuning-approaches)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Results](#results)
8. [How to Run](#how-to-run)
9. [Dependencies](#dependencies)
10. [Acknowledgements](#acknowledgements)

## Background
In dialogue summarization tasks, language models often lack the understanding and intuition needed to generate concise summaries. While methods like few-shot prompting can show improvement in performance, they remain limited in capturing the task's complexity.

Previously, I had done a project using prompt engineering suhc as few-shots prompting and generation configuring techniques to improve the model response with small achievement, there are rooms to be improved.

This project addresses these challenges by fine-tuning the `google/flan-t5-base` model. Fine-tuning adjusts the model's weights to specialize it for specific tasks, allowing for better interpretability, style control, and improved performance.

## Project Objectives
- Fine-tune the `google/flan-t5-base` model to improve its dialogue summarization capabilities.
- Explore full fine-tuning and Parameter-Efficient Fine-Tuning (PEFT) techniques, such as **LoRA** (Low-Rank Adaptation), to reduce computational demands.
- Evaluate the model performance quantitatively with ROUGE metrics.

## Dataset
This project uses the DialogSum dataset from [HuggingFace datasets](https://huggingface.co/datasets/knkarthick/dialogsum).
The DialogSum dataset is a collection of 13,460 dialogues primarily focused on open-domain conversations. Each dialogue is complemented by a concise manually-written summary. The dataset is divided into three splits:
- **Train split**: Used for model training.
- **Validation split**: Used for hyperparameter tuning and monitoring training.
- **Test split**: Used for evaluating the model's performance on unseen data.

This dataset is highly suitable for training summarization models, as it provides clear task-specific ground truth (e.g., concise summaries of multi-turn conversations involving multiple participants).

### Dataset Fields:
1. **ID**: A unique identifier for each dialogue.
2. **Topic**: The general topic or theme of the conversation.
3. **Dialogue**: The multi-turn conversation text, written in a structured "#Person1#: ..." format.
4. **Summary**: A concise text summary of the corresponding dialogue.

## Data Preparation
The dataset is processed to convert dialogues and their corresponding summaries into structured input-output pairs for training. 

### Preprocessing Steps:
1. **Formatting Input Prompts**:
   Each dialogue is reformatted into an instruction-like prompt, making it explicit for the model to know the task. Example template:
   ```
   Summarize the following conversation.

   #Person1#: [Dialogue]
   #Person2#: [Dialogue]
   ...
   Summary:
   ```
   This aligns with the input design used for fine-tuning.

2. **Tokenization**:
   - Both the input prompts (dialogues) and labels (summaries) are tokenized using the [HuggingFace tokenizer](https://huggingface.co/docs/transformers/v4.0.0/en/tokenizer_summary).
   - The tokens are padded or truncated to a maximum sequence length to account for the model's fixed input window.

3. **Dataset Trimming**:
   - To optimize for computational resources, only a subset of the dataset was used for training. I filtered out every 100th example for quicker experimentation:
   ```python
   tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)
   ```

### Example Preprocessed Data:
- **Input Prompt (Dialogue)**:
  ```
  Summarize the following conversation.
  
  #Person1#: Have you considered switching to solar energy?
  #Person2#: I have, but it's quite expensive in the beginning.
  #Person1#: True, but it saves a lot of money in the long run.
  ...
  
  Summary:
  ```
- **Label (Summary)**:
  ```
  #Person1# discusses the benefits of solar energy with #Person2#.
  ```

  
## Fine-Tuning Approaches

### 1. Full Fine-Tuning
This involves fine-tuning all model parameters for the task. While computationally expensive, it can yield performance improvements on all evaluation metrics.

### 2. Parameter-Efficient Fine-Tuning (PEFT)
PEFT, specifically **LoRA**, introduces trainable adapter layers into the model. When combined with the pretrained model, these adapters significantly reduce training costs while achieving similar performance.

- **Advantages of PEFT**:
  - Lower computational and memory requirements.
  - Modular: Fine-tuned adapters can be easily shared or reused without altering the original base model.

## Evaluation Metrics
To evaluate the summarization performance, this project uses the [ROUGE metric](https://en.wikipedia.org/wiki/ROUGE_(metric)), which measures text similarity based on n-gram and sequence overlap:
- **ROUGE-1** (Unigram overlap)
- **ROUGE-2** (Bigram overlap)
- **ROUGE-L** (Longest common subsequence)

These metrics help compare summaries generated by the models to reference human summaries.

## Results
Experiments demonstrate the following:
1. Fully fine-tuned models outperform zero-shot summarization significantly.
2. PEFT models achieve near-parity performance with fully fine-tuned models while requiring significantly fewer resources.
3. Overall, PEFT offers an effective balance between cost and performance.

Sample Improvements:
- The fully fine-tuned model achieved consistent improvements across ROUGE-1, ROUGE-2, and ROUGE-L metrics.
- PEFT models improved computational efficiency with only a minor performance tradeoff compared to full fine-tuning.

## How to Run
Follow these steps to replicate the project:

1. **Install Dependencies**
   Make sure you have Python 3.9 or later installed. Clone the repository and install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Notebook**
   Open the Jupyter notebook included in this repository to explore the fine-tuning and evaluation steps:
   ```bash
   jupyter notebook dialogue_summaries_with_fine_tuning.ipynb
   ```

3. **Experiment with Parameters**
   You can adjust training parameters (e.g., number of epochs, learning rate) to further explore the performance trade-offs between fine-tuning techniques.

## Dependencies
This project relies on the following major libraries:
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [Datasets](https://huggingface.co/docs/datasets/)
- [PEFT](https://github.com/huggingface/peft)
- [Evaluate](https://huggingface.co/spaces/evaluate-metric)
- [PyTorch](https://pytorch.org/)
- Pandas, NumPy, Jupyter Notebook

See `requirements.txt` for the full list of dependencies.

## Acknowledgements
This project leverages:
- The [DialogSum dataset](https://huggingface.co/datasets/knkarthick/dialogsum) for dialogue summarization tasks.
- The [google/flan-t5-base](https://huggingface.co/google/flan-t5-base) model.
- HuggingFace's tools for fine-tuning and evaluation.

Feel free to explore, experiment, and improve the methods in this repository. Contributions are welcome!
