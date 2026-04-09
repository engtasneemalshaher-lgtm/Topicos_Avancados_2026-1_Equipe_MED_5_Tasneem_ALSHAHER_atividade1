# LLM Benchmarking on Medical QA – Tasneem Alshaher

**Universidade Federal de Sergipe (UFS)**  
**Course:** Tópicos Avançados em Computação  
**Instructor:** Prof. Glauco Carneiro  
**Term:** 2026.1  
**Team:** 5 – Medical Domain  
**Student:** Tasneem Alshaher — Registration: 202611011450

---

## About This Work

This is my individual contribution to Activity 1 of the course. The goal was to select a set of medical questions, run them through three different language models, and measure how well each model performed — both on open questions with expert-written gold answers and on multiple-choice questions with a known correct option.

My assigned question blocks:
- **Open-ended:** Questions 118–134 (K-QA dataset)
- **Multiple Choice:** Questions 190–216 (USMLE dataset)

---

## Data Sources

The activity used two datasets from the Medical domain (Team 5):

**M1 – Open-ended questions:**  
K-QA dataset by Itay Manes et al., available at:  
https://github.com/Itaymanes/K-QA/blob/main/dataset/questions_w_answers.jsonl

Each entry contains a `Question`, a `Free_form_answer` written by a medical specialist, a `Must_have` list of required concepts, and additional metadata. The reference paper is:  
> Manes et al., *K-QA: A Real-World Medical Q&A Benchmark*, BioNLP @ ACL 2024.

**M2 – Multiple-choice questions:**  
USMLE-style questions with answer key (27 questions from my assigned range).

---

## Models

I chose three models that differ in size, architecture, and training focus:

| # | Model | Size | Notes |
|---|-------|------|-------|
| 1 | **Llama-3-8B-Instruct** | 8B | Meta's instruction-tuned Llama 3 |
| 2 | **Phi-3-mini-4k-instruct** | ~3.8B | Microsoft's compact but capable model |
| 3 | **TinyLlama-1.1B-Chat** | 1.1B | Lightweight; included to test small-model limits |

All three were loaded with 4-bit NF4 quantization using `bitsandbytes` on a Colab T4 GPU. Chat templates were applied where available (`apply_chat_template`), so each model received prompts in its expected format.

---

## Evaluation Design

### Open-ended scoring
Since the K-QA dataset provides a `Must_have` field — a list of concepts that a correct answer must contain — I used that as the scoring signal. The score for each response is the fraction of must-have terms found in the model's output:

```
coverage = matched_terms / total_must_have_terms
```

A term is considered matched if at least half its words appear in the response (case-insensitive). Scores range from 0 to 1.

### MCQ scoring
Models were prompted to output a single letter. After extracting the predicted letter using pattern matching, I compared it to the gold answer. The final metric is **accuracy**.

---

## Results Summary

### Multiple-Choice Accuracy

| Model | Correct / Total | Accuracy |
|-------|-----------------|----------|
| Phi-3-mini | 15 / 27 | **55.6%** |
| Llama-3-8B | 13 / 27 | **48.1%** |
| TinyLlama-1.1B | 6 / 27 | **22.2%** |

### Open-ended Must-Have Coverage

| Model | Mean | Median | Std Dev | Answers ≥ 0.5 | Answers ≥ 0.8 |
|-------|------|--------|---------|----------------|----------------|
| Llama-3-8B | **0.574** | 0.578 | 0.287 | 11 | 5 |
| Phi-3-mini | 0.516 | 0.550 | 0.244 | 9 | 3 |
| TinyLlama-1.1B | 0.431 | 0.444 | 0.230 | 8 | 0 |

---

## Discussion

The results reveal an interesting pattern: no single model dominates across both tasks.

**Phi-3-mini** was the best MCQ performer by a notable margin (55.6%), which is impressive for a model under 4B parameters. Its structured, concise outputs seem well-suited to the multiple-choice format.

**Llama-3-8B** led on open-ended questions, with the highest mean coverage (0.574) and the most answers scoring above 0.8. Its larger capacity allows it to generate richer, more complete responses that capture more of the required medical concepts.

**TinyLlama** was clearly the weakest overall. On MCQ it scored only 22.2% — barely above random for a 6-option task — and its open-ended responses never achieved the excellent (≥0.8) threshold, showing that 1.1B parameters are insufficient for reliable clinical reasoning.

A key observation: MCQ and open-ended tasks measure different things. Phi-3 excels at selection but Llama-3 generates more informative free-form answers. For real medical applications, both abilities matter, and a model that balances them well would be ideal.

---

## Files in This Folder

```
.
├── README_Tasneem.md                          ← this file
├── tasneem_Llama-3-8B-Instruct.ipynb         ← Llama 3 inference notebook
├── tasneem_Phi-3-mini-4k-instruct.ipynb      ← Phi-3 inference notebook
├── tasneem_TinyLlama-1_1B-Chat.ipynb         ← TinyLlama inference notebook
├── llama3_summary.csv                         ← Llama 3 results
├── phi3_summary.csv                           ← Phi-3 results
└── tinyllama_summary.csv                      ← TinyLlama results
```

---

## Replication Instructions

Each notebook is self-contained and designed for Google Colab (T4 GPU):

1. Open the notebook in Colab and enable GPU runtime.
2. Mount your Google Drive and update the path constants at the top of the notebook.
3. Place the dataset files (`questions_w_answers.jsonl`, MCQ CSVs) in the expected Drive location.
4. Run all cells from top to bottom. Summary CSVs will be saved automatically.

Required packages (installed within each notebook):
```bash
pip install transformers accelerate bitsandbytes
```

For Llama-3, a Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct` is required (gated model). Log in with `huggingface-cli login` or set your token in Colab secrets.

