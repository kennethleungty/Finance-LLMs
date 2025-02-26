<h1 align="center"><strong>📈 🧠<br>Finance LLMs</strong></h1>
<h2 align="center"><strong>Comprehensive Compilation of LLMs for Financial Services</strong></h2>


## 📌 Context  
Large Language Models (LLMs) have revolutionized natural language processing, demonstrating exceptional capabilities across diverse applications—from financial analysis to risk assessment and automated reporting.

As financial institutions recognize the value of LLMs, there is a growing trend toward customizing models for financial services, ensuring they capture sector-specific expertise, regulations, and terminology.

This repository serves as a centralized collection of finance-focused LLMs, documenting how companies and research groups develop specialized models tailored for banking, asset management, trading, risk analysis, and regulatory compliance.

By bridging the gap between LLMs and financial applications, this collection showcases real-world implementations, helping track advancements and trends in AI-driven financial services.

If you know of an finance industry-specific LLM that should be added to this repository, feel free to submit a **pull request** or open an **issue**! 
More info in link below:  

[![Contributions Welcome!](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](./CONTRIBUTING.md)
___
## 📑 Contents
- [Banking & Payments](#banking)
    - Encompasses retail and commercial banking, central banking, digital payments, lending, and credit services.
- [Investments & Capital Markets](#investments)
    - Includes asset management, wealth management, stock and bond markets, trading, brokerage, private equity, and fintech innovations.
- [Insurance & Risk Management](#insurance)
    - Covers life, health, property, casualty insurance, and broader risk management solutions.
___
## 📚 Compilation  
<!-- Copy the following string to create a new entry! -->
<!-- | LLM Name | Training Type (e.g., Fine-tuned) | Month Year | Brief description | [🔗](https://github_or_website.com) | [🔗](https://arxiv_or_other_paper.com) | -->

<a name="banking"></a>
## Banking & Payments
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| CommBiz Gen AI | Undisclosed | Jan 2025 | Together with AWS, the Commonwealth Bank of Australia (CBA) rolled out an AI tool to assist tens of thousands of business customers with inquiries, facilitating quicker payments and efficient transactions. | [🔗](https://au.finance.yahoo.com/news/commonwealth-bank-launches-new-ai-tool-to-reimagine-banking-faster-and-safer-005744886.html) | - |
| North for Banking | Pre-trained | Jan 2025 | RBC and Cohere co-developed and securely deployed an enterprise generative AI (genAI) solution optimized for financial services, building upon Cohere's proprietary foundation models | [🔗](https://www.rbc.com/newsroom/news/article.html?article=125967)  | - |
| BBVA & OpenAI | Pre-trained | Nov 2024 | BBVA signed an agreement with OpenAI for 3,000 ChatGPT Enterprises licenses, leading to increased productivity and creativity. Staff across various departments have developed over 2,900 specialized GPTs for tasks such as translating risk-specific terminology and drafting responses to client inquiries. | [🔗](https://www.wsj.com/articles/six-months-thousands-of-gpts-and-some-big-unknowns-inside-openais-deal-with-bbva-5d6f1c03?utm_source=chatgpt.com) | - |
| Bitext Mistral-7b-Banking | Fine-tuned | Jun 2024 | Fine-tuned version of the Mistral-7B-Instruct-v0.2, specifically tailored for the banking domain. It is optimized to answer questions and assist users with various banking transactions | [🔗](https://github.com/bitext/bitext-mistral-7b-banking) | [🔗](https://www.bitext.com/blog/general-purpose-models-verticalized-enterprise-genai/) |
| IDEA-FinQA | Agentic RAG | Jun 2024 | Financial question-answering system based on Qwen1.5-14B-Chat, utilizing real-time knowledge injection and supporting various data collection and querying methodologies, and comprises three main modules: the data collector, the data querying module, and LLM-based agents tasked with specific functions. | [🔗](https://github.com/IDEA-FinAI/IDEAFinBench) | [🔗](https://arxiv.org/abs/2407.00365) | 
| Ask FT | Undisclosed | Mar 2024 | LLM tool by Financial Times (FT) that enables subscribers to query and receive responses derived from two decades of published FT content. | [🔗](https://aboutus.ft.com/press_release/financial-times-launches-first-generative-ai-tool) | - |
| RAVEN | Fine-tuned | Jan 2024 | Fine-tuned LLaMA-2 13B Chat model designed to enhance financial data analysis by integrating external tools. Used supervised fine-tuning with parameter-efficient techniques, utilizing a diverse set of financial question-answering datasets, including TAT-QA, Financial PhraseBank, WikiSQL, and OTT-QA | - | [🔗](https://arxiv.org/abs/2401.15328) |
| FinMA | Fine-tuned | Jun 2023 | Comprehensive framework that introduces FinMA (Financial Multi-task Assistant), an open-source financial LLM fine-tuned (7B and 30B versions) from LLaMA using a diverse, multi-task instruction dataset of 136,000 samples. The dataset encompasses various financial tasks, document types, and data modalities. | [🔗](https://github.com/chancefocus/PIXIU) | [🔗](https://arxiv.org/abs/2306.05443) |
| XuanYuan 2.0 | Pre-trained & Fine-tuned | May 2023 | Chat model (built upon the BLOOM-176B architecture) trained by combining general-domain with domain-specific knowledge and integrating the stages of pre-training and fine-tuning, It is capable of providing accurate and contextually appropriate responses in the Chinese financial domain. | - | [🔗](https://arxiv.org/abs/2305.12002) |
| BBT-FinT5 | Pre-trained | Feb 2023 | Chinese financial pre-training language model (1B parameters) based on the T5 model, and pre-trained on the 300Gb financial corpus called FinCorpus | - | [🔗](https://arxiv.org/pdf/2302.09432) |


___
<a name="investments"></a>
## Investments & Capital Markets
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| TigerGPT | Pre-trained | Feb 2025 | Tiger Brokers integrated DeepSeek's AI model, DeepSeek-R1, into its AI-powered chatbot, TigerGPT. This adoption aims to enhance market analysis and trading capabilities for its customers through the improved logical reasoning capabilities. | [🔗](https://www.reuters.com/technology/artificial-intelligence/tiger-brokers-adopts-deepseek-model-chinese-brokerages-funds-rush-embrace-ai-2025-02-18) | - |
| FinTral | Pre-trained | Aug 2024 | Suite of multimodal LLMs built upon the Mistral-7b model and tailored for financial analysis. FinTral integrates textual, numerical,  tabular, and image data, and is pretrained on a 20 billion token, high quality dataset | - | [🔗](https://aclanthology.org/2024.findings-acl.774/) |
| JPMorgan Chase IndexGPT | Undisclosed | Jul 2024 | JPMorgan Chase launched a generative AI-based tool (via AWS Bedrock) called IndexGPT, designed to serve as a 'research analyst' for over 50,000 employees, aiding in various tasks that enhance productivity and decision-making within the firm. It is able to generate and refine written documents, provide creative solutions and summarize extensive documents. | [🔗](https://qz.com/jpmorgan-indexgpt-ai-chatbot-investment-advice-1850478529) | - |
| InvestLM | Fine-tuned | Sep 2023 | Financial domain LLM tuned on LLaMA-65B, using a carefully curated instruction dataset related to financial investment. The small yet diverse instruction dataset covers a wide range of financial related topics, from Chartered Financial Analyst (CFA) exam questions to SEC filings to Stackexchange quantitative finance discussions. | [🔗](https://github.com/AbaciNLP/InvestLM) | [🔗](https://arxiv.org/abs/2309.13064) |
| CFGPT | Pre-trained & Fine-tuned | Sep 2023 | Financial LLM based on InternLM-7B that is designed to handle financial texts effectively. It was pre-trained on 584 million documents (141 billion tokens) from Chinese financial sources like announcements, research reports, social media content, and financial news, and then fine-tuned on 1.5 million instruction pairs (1.5 billion tokens) tailored for specific tasks of financial analysis and decision-making. | [🔗](https://github.com/TongjiFinLab/CFGPT) | [🔗](https://arxiv.org/abs/2309.10654)
| FinGPT | Fine-tuned | Jun 2023 | Open-source financial LLM (FinLLM) using a data-centric approach (based on Llama 2) for automated data curation and efficient adaptation, aiming to democratize AI in finance with applications in robo-advising, algorithmic trading, and low-code development. | [🔗](https://github.com/AI4Finance-Foundation/FinGPT) | [🔗](https://arxiv.org/abs/2306.06031) |
| Fin-Llama | Fine-tuned | June 2023 | Specialized version of LLaMA 33B, fine-tuned (with QLoRA and 4-bit quantization) for financial applications using a 16.9k instruction dataset. | [🔗](https://github.com/Bavest/fin-llama) | - |
| Cornucopia-LLaMA-Fin-Chinese | Pre-trained | Apr 2023 | Open-source LLaMA-based model fine-tuned for Chinese financial applications. It uses instruction tuning with Chinese financial Q&A datasets to enhance domain-specific performance. | [🔗](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese) | - |
| BloombergGPT | Pre-trained | Mar 2023 | 50-billion-parameter LLM specifically designed for financial applications and the industry's unique terminology, trained on a 363-billion-token dataset sourced from Bloomberg’s proprietary data, complemented with 345 billion tokens from general-purpose datasets | [🔗](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) | [🔗](https://arxiv.org/abs/2303.17564) |
| Morgan Stanley & OpenAI | Pre-trained | Mar 2023 | Morgan Stanley Wealth Management announced a partnership with OpenAI to develop an internal-facing GPT-4-powered assistant, allowing their financial advisors to query the bank’s vast research repository and internal knowledge base in natural language | [🔗](https://www.morganstanley.com/press-releases/key-milestone-in-innovation-journey-with-openai) | - |
| FLANG-ELECTRA | Pre-trained | Oct 2022 | Domain specific Financial LANGuage model (FLANG) which uses financial keywords and phrases for better masking, and built on the ELECTRA-base architecture. *Note: Considered a smaller LM as it has fewer than 1B params* | [🔗](https://github.com/SALT-NLP/FLANG) | [🔗](https://arxiv.org/abs/2211.00083)
| FinBERT-21 | Pre-trained | Jul 2020 | FinBERT (BERT for Financial Text Mining) is a domain specific language model pre-trained on large-scale financial corpora, allowing it to capture language knowledge and semantic information from the finance domain. *Note: Considered a smaller LM as it has fewer than 1B params* | - | [🔗](https://www.ijcai.org/proceedings/2020/622) |


___
<a name="insurance"></a>
## Insurance & Risk Management
| Name | Type | Date | Description | Website | Paper |
| --- | --- | --- | --- | --- | --- |
| Allianz Insurance Copilot | Undisclosed | Feb 2025 | An internal generative AI system to assist with claims management. Launched in 2024 for auto claims, it leverages cutting-edge LLMs to streamline workflows and automate key tasks. | [🔗](https://www.allianz.com/en/mediacenter/news/articles/250205-smarter-claims-management-smoother-settlements.html) | - |
| AllianzGPT | Undisclosed | Feb 2025 | Internal chatbot that leverages capabilities from different LLM providers (e.g., OpenAI, DeepSeek) on Azure to support general productivity tasks, as well as specific functional areas like audit, Risk Consulting actuarial and other business areas. | [🔗](https://www.allianz.com/en/mediacenter/news/articles/250218-ai-at-allianz-the-impact-of-allianzgpt.html) | - |
| Open-Insurance-LLM-Llama3-8B | Fine-tuned | Nov 2024 | Llama 3 (8B model) fine-tuned with LoRA on the InsuranceQA dataset – a corpus of insurance domain Q&A pairs – to specialize it for insurance queries and conversations | [🔗](https://huggingface.co/Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B/tree/main) | - |
| Zurich Insurance & OpenAI | Pre-trained | Nov 2024 | Zurich Insurance Group turned to Microsoft Azure OpenAI Service to develop AI applications that lead to more accurate and efficient risk management evaluations, accelerating the underwriting process, reducing turnaround times, and increasing customer satisfaction. | [🔗](https://www.microsoft.com/en/customers/story/19760-zurich-insurance-azure-open-ai-service) | - |
| EXL Insurance LLM | Fine-tuned | Sep 2024 | Industry-specific LLM that supports critical claims and underwriting-related tasks, such as claims reconciliation, data extraction and interpretation, question-answering, anomaly detection and chronology summarization. EXL utilized NVIDIA NeMo end-to-end platform for the fine-tuning process on 2 billion tokens of private insurance data. | [🔗](https://www.exlservice.com/about/newsroom/exl-launches-specialized-insurance-large-language-model-leveraging-nvidia-ai-enterprise) | - |
| Swiss Re & mea | Undisclosed | Sep 2024 | Swiss Re partnered with mea (a generative AI-powered global platform) for insurance process automation, such as extracting unstructured data from submission-related documents (e.g., Schedules of Value) and convert it into structured, analysable data. | [🔗](https://www.reinsurancene.ws/swiss-re-selects-mea-platforms-genai-solution-to-enhance-operations/) | - |
| Bitext Mistral-7B-Insurance | Fine-tuned | Jul 2024 | Fine-tuned version of Mistral-7B-Instruct-v0.2, specifically tailored for the insurance domain. It is optimized to answer questions and assist users with various insurance-related procedures | [🔗](https://huggingface.co/bitext/Mistral-7B-Insurance) | [🔗](https://www.bitext.com/blog/general-purpose-models-verticalized-enterprise-genai/) |
| Roots Automation - InsurGPT | Fine-tuned | May 2023 | Roots Automation released an LLM fine-tuned on insurance-specific documents (ACORD forms, First Notice of Loss (FNOL), loss runs) to automate and cut claims processing time significantly | [🔗](https://www.prnewswire.com/news-releases/roots-automation-introduces-insurgpt---the-worlds-most-advanced-generative-ai-model-for-insurance-301823620.html) | - |
