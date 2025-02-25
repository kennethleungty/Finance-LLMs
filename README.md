<p align="center" style="font-size: 50px;">
  🏭 🧠
</p>
<p align="center" style="font-size: 40px; font-weight: bold;">
  Industry-Specific LLMs
</p>
<p align="center" style="font-size: 30px; font-weight: bold;">
  Comprehensive Compilation of LLMs Tailored for Specific Industries & Domains  
</p>

# 📌 Context  
Large Language Models (LLMs) have transformed natural language processing, demonstrating exceptional capabilities across diverse applications—from text generation to complex problem-solving.  

As organizations recognize the value of LLMs, there is a growing trend toward **customizing models for specific industries**, ensuring they capture sector-specific expertise, terminology, and nuances.  

This repository serves as a **centralized collection of industry-specific LLMs**, documenting how companies and research groups develop specialized models tailored for fields such as **finance, healthcare, law, entertainment, and beyond**.  

By bridging the gap between LLMs and highly specialized applications, this collection showcases **real-world implementations**, helping track advancements and trends in industry-driven LLM development.  

If you know of an industry-specific LLM that should be added to this repository, feel free to submit a **pull request** or open an **issue**! 
More info in link below:  

[![Contributions Welcome!](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](./CONTRIBUTING.md)
___
# 📑 Contents
- [Finance](#finance)
- [Healthcare](#healthcare)
- [Information Technology](#it)
- [Science](#science)
- [Telecommunications](#telco)
___
# 📚 Compilation  
<!-- Copy the following string to create a new entry! -->
<!-- | LLM Name | Training Type (e.g., Fine-tuned) | Month Year | Brief description | [🔗](https://github_or_website.com) | [🔗](https://arxiv_or_other_paper.com) | -->

<a name="finance"></a>
## Finance
| Name | Type | Date | Description | Website/Repo | Paper |
| --- | --- | --- | --- | --- | --- |
| FinTral | Pre-trained | Aug 2024 | Suite of multimodal LLMs built upon the Mistral-7b model and tailored for financial analysis. FinTral integrates textual, numerical,  tabular, and image data, and is pretrained on a 20 billion token, high quality dataset | - | [🔗](https://aclanthology.org/2024.findings-acl.774/) |
| IDEA-FinQA | Agentic RAG | Jun 2024 | Financial question-answering system based on Qwen1.5-14B-Chat, utilizing real-time knowledge injection and supporting various data collection and querying methodologies, and comprises three main modules: the data collector, the data querying module, and LLM-based agents tasked with specific functions. | [🔗](https://github.com/IDEA-FinAI/IDEAFinBench) | [🔗](https://arxiv.org/abs/2407.00365) | 
| Ask FT | Undisclosed | Mar 2024 | LLM tool by Financial Times (FT) that enables subscribers to query and receive responses derived from two decades of published FT content. | [🔗](https://aboutus.ft.com/press_release/financial-times-launches-first-generative-ai-tool) | - |
| RAVEN | Fine-tuned | Jan 2024 |  Fine-tuned LLaMA-2 13B Chat model designed to enhance financial data analysis by integrating external tools. Used supervised fine-tuning with parameter-efficient techniques, utilizing a diverse set of financial question-answering datasets, including TAT-QA, Financial PhraseBank, WikiSQL, and OTT-QA | - | [🔗](https://arxiv.org/abs/2401.15328) |
| FinMA | Fine-tuned | Jun 2023 | Comprehensive framework that introduces FinMA, an open-source financial LLM fine-tuned from LLaMA using a diverse, multi-task instruction dataset of 136,000 samples. The dataset encompasses various financial tasks, document types, and data modalities. | [🔗](https://github.com/chancefocus/PIXIU) | [🔗](https://arxiv.org/abs/2306.05443) |
| FinGPT | Fine-tuned | Jun 2023 | Open-source financial large language model (FinLLM) using a data-centric approach (based on Llama 2) for automated data curation and efficient adaptation, aiming to democratize AI in finance with applications in robo-advising, algorithmic trading, and low-code development. | [🔗](https://github.com/AI4Finance-Foundation/FinGPT) | [🔗](https://arxiv.org/abs/2306.06031) |
| BloombergGPT | Pre-trained | Mar 2023 | 50-billion-parameter Large Language Model (LLM) specifically designed for financial applications, trained on a 363-billion-token dataset sourced from Bloomberg’s proprietary data, complemented with 345 billion tokens from general-purpose datasets | [🔗](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) | [🔗](https://arxiv.org/abs/2303.17564) |


___
<a name="healthcare"></a>
## Healthcare
| Name | Type | Date | Description | Website/Repo | Paper |
| --- | --- | --- | --- | --- | --- |
| PH-LLM | Fine-tuned | Jun 2024 | The Personal Health Large Language Model (PH-LLM) is a fine-tuned version of Gemini, designed to generate insights and recommendations to improve personal health behaviors related to sleep and fitness patterns. | [🔗](https://research.google/blog/advancing-personal-health-and-wellness-insights-with-ai/) | [🔗](https://arxiv.org/abs/2406.06474) |
| RUSSELL-GPT | Fine-tuned | Aug 2023 | LLM developed by National University Health System in Singapore to enhance clinicians' productivity (e.g., medical Q&A, case note summarization) | [🔗](https://www.nuhsplus.edu.sg/article/ai-healthcare-in-nuhs-receives-boost-from-supercomputer) | - | 
| PharmacyGPT | In-context Learning | Jul 2023 | Framework based on LLMs like GPT-4 in clinical pharmacy roles, utilizing dynamic prompting and iterative optimization to enhance performance in tasks such as patient clustering, medication planning, and outcome prediction, based on ICU data from University of North Carolina Health System | - | [🔗](https://arxiv.org/abs/2307.10432) |
| Med-PaLM 2 | Fine-tuned | Jul 2023 | Medical LLM based on PaLM 2, fine-tuned with medical domain adaptation. Training involved instruction tuning on a mix of curated medical datasets, including research papers, medical licensing exam questions, clinical data, and expert-annotated medical dialogues.  | [🔗](https://cloud.google.com/blog/topics/healthcare-life-sciences/sharing-google-med-palm-2-medical-large-language-model) | [🔗](https://arxiv.org/pdf/2305.09617.pdf) |
| Med-PaLM | Fine-tuned | Dec 2022 | Google's LLM (fine-tuned using PaLM as base model) designed to provide high quality answers to medical questions. | [🔗](https://sites.research.google/med-palm/) | [🔗](https://www.nature.com/articles/s41586-023-06291-2)  |


<a name="science"></a>
## Science
| Name | Type | Date | Description | Website/Repo | Paper |
| --- | --- | --- | --- | --- | --- |
| ProtGPT2 | Pre-trained | Jul 2022 |  LLM (with 738 million parameters) specifically for protein engineering and design by being trained on the protein space that generates de novo protein sequences following principles of natural ones. | [🔗](https://huggingface.co/nferruz/ProtGPT2) | [🔗](https://www.nature.com/articles/s41467-022-32007-7) |

___
<a name="it"></a>
## Information Technology (IT)
| Name | Type | Date | Description | Website/Repo | Paper |
| --- | --- | --- | --- | --- | --- |
| OWL | Fine-tuned | Sep 2023 | specialized Large Language Model (LLM) designed for IT operations, trained on the Owl-Instruct dataset, which encompasses nine domains such as information security, system architecture, and databases. | [🔗](https://arxiv.org/abs/2309.09298) | - |

___
<a name="telco"></a>
## Telecommunications
| Name | Type | Date | Description | Website/Repo | Paper |
| --- | --- | --- | --- | --- | --- |
| TelecomGPT | Fine-tuned | Jul 2024 | Domain-specific LLM for telecommunications, fine-tuned through continual pre-training, instruction tuning, and alignment tuning on telecom datasets | - | [🔗](https://arxiv.org/abs/2407.09424) |
