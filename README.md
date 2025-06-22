
# Synthetic Data
As AI systems become increasingly data-hungry, the need for high-quality datasets has never been greater. However, real-world data collection faces major challenges: scarcity, privacy constraints, and high acquisition costs. Synthetic Data Generation (SDG) offers a compelling alternative—creating artificial data that mimics real-world patterns without the associated drawbacks.

> 📌 This repository complements the survey [*A Survey of LLM-Based Methods for Synthetic Data Generation and the Rise of Agentic Workflows*](https://link.springer.com/chapter/10.1007/978-3-031-93418-6_9) by Ahmad Alismail and Carsten Lanquillon. It is a continuously updated resource collecting references on LLM-based synthetic data generation—supporting ongoing learning and collaboration in the research community.  
> 💡 If you’d like to contribute or suggest additions, feel free to open a pull request or issue!


## Table of Contents

- [Surveys](#surveys)
- [SDG Methods](#sdg-methods)
  - [Traditional Architectures: Single LLM without External Tools](#traditional-architectures-single-llm-without-external-tools)
  - [Agentic Workflows](#agentic-workflows)
- [Further Reading](#further-reading)
- [Related Repositories](#related-repositories)

# Surveys

| Publication                                                                                                                                          | Date    |
| ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| [Synthetic Data Generation Using Large Language Models: Advances in Text and Code](https://arxiv.org/pdf/2503.14023)                                 | 03-2025 |
| [Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation](https://arxiv.org/abs/2502.17521) | 02-2025 |
| [A Survey on Data Synthesis and Augmentation for Large Language Models](https://arxiv.org/abs/2410.12896)                                            | 10-2024 |
| [Data Augmentation Using LLMs: Data Perspectives, Learning Paradigms, and Challenges](https://aclanthology.org/2024.findings-acl.97/)                | 08-2024 |
| [On LLMs- Driven Synthetic Data Generation, Curation, and Evaluation: A Survey](https://arxiv.org/abs/2406.15126)                                    | 06-2024 |
| [Best Practices and Lessons Learned on Synthetic Data](https://arxiv.org/abs/2404.07503)                                                             | 04-2024 |
| [A Survey on Data Augmentation in the Large Model Era](https://arxiv.org/abs/2401.15422)                                                             | 01-2024 |
| [Comprehensive Exploration of Synthetic Data Generation: A Survey](https://arxiv.org/abs/2401.02524)                                                 | 01-2024 |

# SDG Methods
## Traditional Architectures: Single LLM without External tools

| **Publication**                                                                                                                                                                                                           | **Date** |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| [O1 Replication Journey – Part 2: Surpassing O1-preview through Simple Distillation Big Progress or Bitter Lesson?](https://arxiv.org/abs/2411.16489)                                                                     | 11-2024  |
| [Self-Judge: Selective Instruction Following with Alignment Self-Evaluation](https://arxiv.org/abs/2409.00935)                                                                                                            | 09-2024  |
| [Automated test generation to evaluate tool-augmented LLMs as conversational AI agents](https://arxiv.org/abs/2409.15934)                                                                                                 | 09-2024  |
| [Is Child-Directed Speech Effective Training Data for Language Models?](https://arxiv.org/abs/2408.03617)                                                                                                                 | 08-2024  |
| [Self-Translate-Train: Enhancing Cross-Lingual Transfer of Large Language Models via Inherent Capability](https://arxiv.org/abs/2407.00454)                                                                               | 07-2024  |
| [Case2Code: Learning Inductive Reasoning with Synthetic Data](https://arxiv.org/abs/2407.12504)                                                                                                                           | 07-2024  |
| [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094)                                                                                                                           | 06-2024  |
| [Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)                                                                                                  | 06-2024  |
| [DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data](https://arxiv.org/abs/2405.14333)                                                                                                 | 05-2024  |
| [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)                                                                                                                                                                | 04-2024  |
| [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://proceedings.neurips.cc/paper_files/paper/2024/hash/3d5aa9a7ce28cdc710fbd044fd3610f3-Abstract-Datasets_and_Benchmarks_Track.html)              | 02-2024  |
| [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)                                                                                                         | 01-2024  |
| [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)                                                                                                         | 01-2024  |
| [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)                                                                                                     | 12-2023  |
| [Orca 2: Teaching Small Language Models How to Reason](https://arxiv.org/abs/2311.11045)                                                                                                                                  | 11-2023  |
| [Knowledge-Infused Prompting: Assessing and Advancing Clinical Text Data Generation with Large Language Models](https://arxiv.org/abs/2311.00287)                                                                         | 11-2023  |
| [HELPSTEER: Multi-attribute Helpfulness Dataset for STEERLM](https://arxiv.org/abs/2311.09528)                                                                                                                            | 11-2023  |
| [CoAnnotating: Uncertainty-Guided Work Allocation between Human and Large Language Models for Data Annotation](https://arxiv.org/abs/2310.15638)                                                                          | 10-2023  |
| [ULTRAFEEDBACK: Boosting Language Models with Scaled AI Feedback](https://arxiv.org/abs/2310.01377)                                                                                                                       | 10-2023  |
| [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)                                                                                                                               | 09-2023  |
| [MAMMOTH: BUILDING MATH GENERALIST MODELS THROUGH HYBRID INSTRUCTION TUNING](https://arxiv.org/abs/2309.05653)                                                                                                            | 09-2023  |
| [METAMATH: BOOTSTRAP YOUR OWN MATHEMATICAL QUESTIONS FOR LARGE LANGUAGE](https://arxiv.org/abs/2309.12284)                                                                                                                | 09-2023  |
| [DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation](https://arxiv.org/abs/2308.14346)                                                                                               | 08-2023  |
| [AlpaGasus: Training A Better Alpaca with Fewer Data](https://arxiv.org/abs/2307.08701)                                                                                                                                   | 07-2023  |
| [BEAVERTAILS: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4dbb61cb68671edc4ca3712d70083b9f-Abstract-Datasets_and_Benchmarks.html) | 07-2023  |
| [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)                                                                                                                                                            | 06-2023  |
| [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568)                                                                                                                 | 06-2023  |
| [Orca: Progressive Learning from Complex Explanation Traces of GPT-4](https://arxiv.org/abs/2306.02707)                                                                                                                   | 06-2023  |
| [ToolCoder: Teach Code Generation Models to use API search tools](https://arxiv.org/abs/2305.04032)                                                                                                                       | 05-2023  |
| [TinyStories: how Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759)                                                                                                       | 05-2023  |
| [Can ChatGPT Reproduce Human-Generated Labels? A Study of Social Computing Tasks](https://arxiv.org/abs/2304.10145)                                                                                                       | 04-2023  |
| [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)                                                                                                             | 04-2023  |
| [Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data](https://arxiv.org/abs/2304.01196)                                                                                                    | 04-2023  |
| [HuaTuo: Tuning LLaMA Model with Chinese Medical Knowledge](https://arxiv.org/abs/2304.06975)                                                                                                                             | 04-2023  |
| [Instruction Tuning with GPT-4](https://arxiv.org/abs/2304.03277)                                                                                                                                                         | 04-2023  |
| [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                                                                              | 03-2023  |
| [CHATGPT OUTPERFORMS CROWD WORKERS FOR TEXT-ANNOTATION TASKS](https://www.pnas.org/doi/abs/10.1073/pnas.2305016120)                                                                                                       | 03-2023  |
| [AugGPT: Leveraging ChatGPT for Text Data Augmentation](https://ieeexplore.ieee.org/abstract/document/10858342/)                                                                                                          | 02-2023  |
| [SELF-INSTRUCT: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)                                                                                                              | 12-2022  |
| [CORE: A Retrieve-then-Edit Framework for Counterfactual Data Generation](https://arxiv.org/abs/2210.04873)                                                                                                               | 12-2022  |
| [DISCO: Distilling Counterfactuals with Large Language Models](https://arxiv.org/abs/2212.10534)                                                                                                                          | 12-2022  |
| [STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning](https://research.google/pubs/star-self-taught-reasoner-bootstrapping-reasoning-with-reasoning/)                                                      | 03-2022  |
## Agentic Workflows


| Publication                                                                                                                                     | Date    |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| [Benchmarking Agentic Workflow Generation](https://arxiv.org/abs/2410.07869)                                            | 10-2024 |
| [BenchAgents: Automated Benchmark Creation With Agent Interaction](https://arxiv.org/abs/2410.22584)                                            | 10-2024 |
| [The Fellowship of the LLMs: Multi-Agent Workflows for Synthetic Preference Optimization Dataset Generation](https://arxiv.org/abs/2408.08688)  | 08-2024 |
| [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502)                                                | 07-2024 |
| [Arena Learning: Build Data Flywheel for LLMs Post-Training via Simulated Chatbot Arena](https://arxiv.org/abs/2407.10627)                      | 07-2024 |
| [MALLM-GAN: Multi-Agent Large Language Model as Generative Adversarial Network for Synthesizing Tabular Data](https://arxiv.org/abs/2406.10521) | 06-2024 |
| [Advancing LLM Reasoning Generalists With Preference Trees](https://arxiv.org/abs/2404.02078)                                                   | 04-2024 |
| [LAB: Large-Scale Alignment for Chatbots](https://arxiv.org/abs/2403.01081)                                                                     | 03-2024 |
| [Benchmark self-evolving: A multi-agent framework for dynamic llm evaluation](https://arxiv.org/abs/2402.11443)                                 | 02-2024 |
| [Synthetic data (almost) from scratch: Generalized instruction tuning for language models](https://arxiv.org/abs/2402.13064)                    | 02-2024 |
| [Orca-math: Unlocking the potential of SLMs in grade school math](https://arxiv.org/abs/2402.14830)                                             | 02-2024 |
| [Learning From Mistakes Makes LLM a Better Reasoner](https://arxiv.org/abs/2310.20689)                                                          | 10-2023 |

# Further Reading


| Publication                                                                                                             | Date    | Notes                                                                                                                                                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Evaluating Language Models as Synthetic Data Generators](https://arxiv.org/abs/2412.03679)                             | 12-2024 | a benchmark that evaluates LLMs’ abilities to generate synthetic data by comparing outputs from multiple models and analyzing quality metrics (e.g., perplexity, difficulty), revealing that data-generation prowess doesn’t always match problem-solving strength |
| [On the Diversity of Synthetic Data and its Impact on Training Large Language Models](https://arxiv.org/abs/2410.15226) | 10-2024 | Introduces a diversity metric (“LLM cluster-agent”) to quantify synthetic data variety, demonstrating that data diversity boosts model performance—especially during fine-tuning—even for smaller-scale LLMs                                                       |
| [Scaling Laws of Synthetic Data for Language Models](https://arxiv.org/abs/2503.19551)                                  | 03-2025 | Presents _SynthLLM_, a framework revealing that synthetic pre-training data follows power-law scaling up to 300B tokens, and larger LLMs require fewer synthetic tokens to reach optimal performance                                                               |

# Related Repositories

| [Awesome Synthetic Datasets](https://github.com/davanstrien/awesome-synthetic-datasets) | Practical resources for building synthetic text and vision datasets. |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [LLM Synthetic Data](https://github.com/wasiahmad/Awesome-LLM-Synthetic-Data/tree/main) | Papers, tools, and blogs on LLM-generated data.                      |
| [LLM-Datasets](https://github.com/mlabonne/llm-datasets/tree/main):                     | Curated datasets and tools for LLM post-training.                    |

