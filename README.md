<h1 align="center">🏭 🧠</h1>
<h1 align="center"><strong>Industry-Specific LLMs</strong></h1>
<h2 align="center"><strong>Comprehensive Compilation of LLMs Tailored for Specific Industries & Domains</strong></h2>


## 📌 Context  
Large Language Models (LLMs) have transformed natural language processing, demonstrating exceptional capabilities across diverse applications—from text generation to complex problem-solving.  

As organizations recognize the value of LLMs, there is a growing trend toward **customizing models for specific industries**, ensuring they capture sector-specific expertise, terminology, and nuances.  

This repository serves as a **centralized collection of industry-specific LLMs**, documenting how companies and research groups develop specialized models tailored for fields such as **finance, healthcare, law, entertainment, and beyond**.  

By bridging the gap between LLMs and highly specialized applications, this collection showcases **real-world implementations**, helping track advancements and trends in industry-driven LLM development.  

If you know of an industry-specific LLM that should be added to this repository, feel free to submit a **pull request** or open an **issue**! 
More info in link below:  

[![Contributions Welcome!](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](./CONTRIBUTING.md)
___
## 📑 Contents
- [Finance](#finance)
- [Healthcare](#healthcare)
- [Information Technology](#it)
- [Science](#science)
- [Telecommunications](#telco)
___
## 📚 Compilation  
<!-- Copy the following string to create a new entry! -->
<!-- | LLM Name | Training Type (e.g., Fine-tuned) | Month Year | Brief description | [🔗](https://github_or_website.com) | [🔗](https://arxiv_or_other_paper.com) | -->

<a name="finance"></a>
## Finance
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| TigerGPT | Pre-trained | Feb 2025 | Tiger Brokers integrated DeepSeek's AI model, DeepSeek-R1, into its AI-powered chatbot, TigerGPT. This adoption aims to enhance market analysis and trading capabilities for its customers through the improved logical reasoning capabilities. | [🔗](https://www.reuters.com/technology/artificial-intelligence/tiger-brokers-adopts-deepseek-model-chinese-brokerages-funds-rush-embrace-ai-2025-02-18) | - |
| CommBiz Gen AI | Undisclosed | Jan 2025 | Together with AWS, the Commonwealth Bank (CBA) rolled out an AI tool for tens of thousands of its business banking customers, where they can send direct questions to the AI messaging tool and it will provide them with ChatGPT-style responses.  | [🔗](https://au.finance.yahoo.com/news/commonwealth-bank-launches-new-ai-tool-to-reimagine-banking-faster-and-safer-005744886.html) | - |
| North for Banking | Pre-trained | Jan 2025 | RBC and Cohere co-developed and securely deployed an enterprise generative AI (genAI) solution optimized for financial services, building upon Cohere's proprietary foundation models | [🔗](https://www.rbc.com/newsroom/news/article.html?article=125967)  | - |
| FinTral | Pre-trained | Aug 2024 | Suite of multimodal LLMs built upon the Mistral-7b model and tailored for financial analysis. FinTral integrates textual, numerical,  tabular, and image data, and is pretrained on a 20 billion token, high quality dataset | - | [🔗](https://aclanthology.org/2024.findings-acl.774/) |
| JPMorgan Chase LLM Suite | Undisclosed | Jul 2024 | JPMorgan Chase launched a generative AI-based tool (via AWS Bedrock) designed to serve as a ‘research analyst’ for over 50,000 employees, aiding in various tasks that enhance productivity and decision-making within the firm. It is able to generate and refine written documents, provide creative solutions and summarize extensive documents. | [🔗](https://www.cio.com/article/3616622/jpmorgan-chase-builds-ambitious-ai-foundation-on-aws.html) | - |
| IDEA-FinQA | Agentic RAG | Jun 2024 | Financial question-answering system based on Qwen1.5-14B-Chat, utilizing real-time knowledge injection and supporting various data collection and querying methodologies, and comprises three main modules: the data collector, the data querying module, and LLM-based agents tasked with specific functions. | [🔗](https://github.com/IDEA-FinAI/IDEAFinBench) | [🔗](https://arxiv.org/abs/2407.00365) | 
| Ask FT | Undisclosed | Mar 2024 | LLM tool by Financial Times (FT) that enables subscribers to query and receive responses derived from two decades of published FT content. | [🔗](https://aboutus.ft.com/press_release/financial-times-launches-first-generative-ai-tool) | - |
| RAVEN | Fine-tuned | Jan 2024 |  Fine-tuned LLaMA-2 13B Chat model designed to enhance financial data analysis by integrating external tools. Used supervised fine-tuning with parameter-efficient techniques, utilizing a diverse set of financial question-answering datasets, including TAT-QA, Financial PhraseBank, WikiSQL, and OTT-QA | - | [🔗](https://arxiv.org/abs/2401.15328) |
| FinMA | Fine-tuned | Jun 2023 | Comprehensive framework that introduces FinMA, an open-source financial LLM fine-tuned from LLaMA using a diverse, multi-task instruction dataset of 136,000 samples. The dataset encompasses various financial tasks, document types, and data modalities. | [🔗](https://github.com/chancefocus/PIXIU) | [🔗](https://arxiv.org/abs/2306.05443) |
| FinGPT | Fine-tuned | Jun 2023 | Open-source financial large language model (FinLLM) using a data-centric approach (based on Llama 2) for automated data curation and efficient adaptation, aiming to democratize AI in finance with applications in robo-advising, algorithmic trading, and low-code development. | [🔗](https://github.com/AI4Finance-Foundation/FinGPT) | [🔗](https://arxiv.org/abs/2306.06031) |
| BloombergGPT | Pre-trained | Mar 2023 | 50-billion-parameter Large Language Model (LLM) specifically designed for financial applications, trained on a 363-billion-token dataset sourced from Bloomberg’s proprietary data, complemented with 345 billion tokens from general-purpose datasets | [🔗](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) | [🔗](https://arxiv.org/abs/2303.17564) |


___
<a name="healthcare"></a>
## Healthcare
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| PH-LLM | Fine-tuned | Jun 2024 | The Personal Health Large Language Model (PH-LLM) is a fine-tuned version of Gemini, designed to generate insights and recommendations to improve personal health behaviors related to sleep and fitness patterns. | [🔗](https://research.google/blog/advancing-personal-health-and-wellness-insights-with-ai/) | [🔗](https://arxiv.org/abs/2406.06474) |
| Radiology-Llama2 | Fine-tuned | Aug 2023 | Specialized for radiology, and is based on the Llama2 architecture and further trained on a large dataset of radiology reports to generate coherent and clinically useful impressions from radiological findings. | - | [🔗](https://arxiv.org/abs/2309.06419) |
| RUSSELL-GPT | Fine-tuned | Aug 2023 | LLM developed by National University Health System in Singapore to enhance clinicians' productivity (e.g., medical Q&A, case note summarization) | [🔗](https://www.nuhsplus.edu.sg/article/ai-healthcare-in-nuhs-receives-boost-from-supercomputer) | - | 
| PharmacyGPT | In-context Learning | Jul 2023 | Framework based on LLMs like GPT-4 in clinical pharmacy roles, utilizing dynamic prompting and iterative optimization to enhance performance in tasks such as patient clustering, medication planning, and outcome prediction, based on ICU data from University of North Carolina Health System | - | [🔗](https://arxiv.org/abs/2307.10432) |
| Med-PaLM 2 | Fine-tuned | Jul 2023 | Medical LLM based on PaLM 2, fine-tuned with medical domain adaptation. Training involved instruction tuning on a mix of curated medical datasets, including research papers, medical licensing exam questions, clinical data, and expert-annotated medical dialogues.  | [🔗](https://cloud.google.com/blog/topics/healthcare-life-sciences/sharing-google-med-palm-2-medical-large-language-model) | [🔗](https://arxiv.org/pdf/2305.09617.pdf) |
| ChatDoctor | Fine-tuned | June 2023 | Medical chat model fine-tuned on LlaMa using medical domain knowledge. It is based on a large dataset of 100,000 patient-doctor dialogues sourced from a widely used online medical consultation platform | [🔗](https://github.com/Kent0n-Li/ChatDoctor) | [🔗](https://arxiv.org/pdf/2303.14070) |
| Clinical Camel | Fine-tuned | May 2023 | Open-source expert-level medical language model (based on LLaMa-2) with dialogue-based knowledge encoding, and is explicitly tailored for clinical research. 
| Med-PaLM | Fine-tuned | Dec 2022 | Google's LLM (fine-tuned using PaLM as base model) designed to provide high quality answers to medical questions. | [🔗](https://sites.research.google/med-palm/) | [🔗](https://www.nature.com/articles/s41586-023-06291-2)  |


<a name="science"></a>
## Science
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| BioMedLM | Fine-tuned | Mar 2024 | A 2.7B parameter GPT-style autoregressive model trained exclusively on PubMed abstracts and full articles. When fine-tuned, BioMedLM produces strong multiple-choice biomedical question-answering results competitive with much larger models | [🔗](https://huggingface.co/stanford-crfm/BioMedLM) | [🔗](https://arxiv.org/abs/2403.18421) |
| ProtGPT2 | Pre-trained | Jul 2022 |  LLM (with 738 million parameters) specifically for protein engineering and design by being trained on the protein space that generates de novo protein sequences following principles of natural ones. | [🔗](https://huggingface.co/nferruz/ProtGPT2) | [🔗](https://www.nature.com/articles/s41467-022-32007-7) |

___
<a name="it"></a>
## Information Technology (IT)
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| OWL | Fine-tuned | Sep 2023 | specialized Large Language Model (LLM) designed for IT operations, trained on the Owl-Instruct dataset, which encompasses nine domains such as information security, system architecture, and databases. | [🔗](https://arxiv.org/abs/2309.09298) | - |

___
<a name="telco"></a>
## Telecommunications
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| Orange | Pre-trained | Nov 2024 | Orange struck a multi-year partnership with OpenAI that will give the French telecoms operator access to pre-release AI models, and also signed an agreement with Meta and OpenAI to translate regional African languages | [🔗](https://www.reuters.com/technology/artificial-intelligence/orange-signs-deal-with-openai-get-access-pre-release-ai-models-2024-11-27) | - |
| Tele-LLMs | Fine-tuned | Sep 2024 | Open-source LLMs (ranging from 1B to 8B parameters, based on Tinyllama, Gemma, and Llama-3) trained on a comprehensive dataset of telecommunications material curated from relevant sources, and Tele-Eval, a large-scale question-and-answer dataset tailored to the domain.| [🔗](https://arxiv.org/abs/2409.05314) | - |
| TelecomGPT | Fine-tuned | Jul 2024 | Domain-specific LLM for telecommunications, fine-tuned through continual pre-training, instruction tuning, and alignment tuning on telecom datasets | - | [🔗](https://arxiv.org/abs/2407.09424) |
