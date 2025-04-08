<h1 align="center"><strong>📈 🧠<br>Finance LLMs</strong></h1>
<h2 align="center"><strong>Comprehensive Compilation of LLM Implementation in Financial Services</strong></h2>


### 📚 Compilation Context  
Large Language Models are revolutionizing financial services — from banking and trading to compliance and asset management.

This repo is your go-to hub for finance-focused LLMs: curated models, real-world use cases, and the latest AI x Finance innovations driving the industry's next leap forward.

If you know of examples of LLMs used in the finance industry that should be added to this repository, feel free to submit a **pull request** or open an **issue**! 

[![Contributions Welcome!](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](./CONTRIBUTING.md)
___
### 📑 Table of Contents
- [Retail & Commercial Banking](#banking)
    - Covers personal and business banking services—checking and savings accounts, mortgages, SME financing, and credit solutions. Also addresses central banking operations, digital transformation in traditional banks, and AI-driven advisory for both retail and corporate clients.
- [Wealth Management & Capital Markets](#wealth)
    - Encompasses asset and wealth advisory, hedge funds, mutual funds, investment banking, and private equity. Includes securities trading, research, analytics, and AI-driven investment platforms for institutional and retail investors.
- [Payments & FinTech](#payments)
    - Focuses on digital payments, card networks, cross-border transactions, BNPL, embedded finance, and blockchain solutions. Also highlights AI-based fraud detection, risk assessment, and automated payment infrastructure innovations.
- [Insurance & Risk Management](#insurance)
    - Addresses life, health, property, and casualty insurance, along with reinsurance, underwriting automation, and actuarial modeling. Features AI-driven claims processing, fraud detection, personalized policy recommendations, and broader risk assessment tools.

Each example can be classified into one of 3 categories based on type of use case:
- **Enterprise-Wide**: Large-scale LLM deployments spanning multiple lines of business or the entire organization. These implementations typically address a broad range of tasks—such as customer support, internal operations, and compliance—within a unified, centrally managed framework.
- **Specialized Model**: Models either pre-trained or fine-tuned on focused datasets (e.g., regulatory filings, financial news) to achieve deep domain expertise. By narrowing the training data to specific tasks and terminology, they deliver higher accuracy and more relevant insights than general-purpose models.
- **Plug-and-Play**: Quick-to-deploy LLM solutions that handle specific use cases with minimal setup. Financial institutions leverage prompt engineering and ready-made integrations to streamline tasks—such as AI-powered chatbots or document processing—without extensive internal development.
___
<!-- Copy the following string to create a new entry! -->
<!-- | LLM Name | Use Case Type | Month Year | Brief description | [🔗](https://github_or_website.com) | [🔗](https://arxiv_or_other_paper.com) | -->

<a name="banking"></a>
## Retail & Commercial Banking
| Name | Type | Date | Description | Site | Paper |
| --- | --- | --- | --- | --- | --- |
| Capital One Chat Concierge | Specialized LLM | Feb 2025 | Chat Concierge, built on a fine-tuned Llama LLM, helps car buyers compare vehicles, explore financing, estimate trade-in values, and schedule test drives. Fine-tuned on Capital One's proprietary data, it also utilized an agentic approach to understand preferences, tailor recommendations, ensure policy compliance, and engage in human-like interactions. | [🔗](https://www.euromoney.com/article/2eh2s01l11023kmxtleyo/fintech/prem-natarajan-on-capital-ones-ai-stairway-to-heaven#:~:text=Chat%20Concierge%20is%20a%20tool,the%20dealers%E2%80%99%20customer%20relationship%20systems) | - |
| Deutsche Bank & Google | Enterprise-Wide | Feb 2025 | Deutsche Bank leverages Google Cloud’s Vertex AI and Gemini LLMs to streamline banking operation. It involves automating document processing for regulatory compliance, enhancing customer support with AI-powered assistants, and accelerating software development for financial services. | [🔗](https://blog.google/products/google-cloud/deutsche-bank-google-cloud-partner/) | - |
| CommBiz Gen AI | Plug-and-Play | Jan 2025 | Together with AWS, the Commonwealth Bank of Australia (CBA) rolled out a Gen-AI powered messaging service to assist tens of thousands of business customers with inquiries, facilitating quicker payments and efficient transactions. They leveraged Amazon Bedrock Knowledge Bases, Claude 3 and Cohere LLMs, and Amazon OpenSearch as the vector database.| [🔗](https://au.finance.yahoo.com/news/commonwealth-bank-launches-new-ai-tool-to-reimagine-banking-faster-and-safer-005744886.html) | - |
| North for Banking | Enterprise-Wide | Jan 2025 | RBC and Cohere co-developed and securely deployed an enterprise generative AI (genAI) solution optimized for financial services, building upon Cohere's proprietary foundation models | [🔗](https://www.rbc.com/newsroom/news/article.html?article=125967)  | - |
| Banestas & Google | Enterprise-Wide | Dec 2024 | Banestes, a Brazilian bank, used Gemini in Google Workspace to streamline work dynamics, such as accelerating credit analysis by simplifying balance sheet reviews and boosting productivity in marketing and legal departments.  | [🔗](https://workspace.google.com/intl/pt-BR/customers/banestes/) | - |
| Commerzbank & Google | Plug-and-Play | Nov 2024 | Commerzbank utilizes Google's Gemini 1.5 Pro, a multimodal large language model, to automate financial advisory workflows by enabling efficient documentation of client interactions and analysis of complex financial data. Its long context allows it to handle lengthy financial documents and multimedia content seamlessly. | [🔗](https://cloud.google.com/blog/products/ai-machine-learning/how-commerzbank-is-transforming-financial-advisory-workflows-with-gen-ai) | - |
| OCBC ChatGPT | Plug-and-Play | Nov 2024 | OCBC Bank became the first Singapore bank to deploy a generative AI chatbot, OCBC ChatGPT, to all 30,000 employees across 19 countries, aiming to assist with tasks such as writing, research, and ideation. Developed with Microsoft Azure and following a successful six-month trial, the chatbot operates securely on the bank's private cloud to ensure data security. | [🔗](https://www.straitstimes.com/business/ocbc-to-deploy-generative-ai-bot-for-all-30000-staff-globally) | - |
| BBVA & OpenAI | Enterprise-Wide | Nov 2024 | Global financial institution BBVA signed an agreement with OpenAI for 3,000 ChatGPT Enterprises licenses, leading to increased productivity and creativity. Staff across various departments have developed over 2.9k specialized GPTs that boost efficiency, spark creativity, and share expert knowledge across their organization of 125,000 e.g., tasks like translating risk-specific terminology and drafting responses to client inquiries. | [🔗](https://www.wsj.com/articles/six-months-thousands-of-gpts-and-some-big-unknowns-inside-openais-deal-with-bbva-5d6f1c03?utm_source=chatgpt.com) | - |
| PennyMac & Google | Enterprise-Wide | Oct 2024 | Pennymac Financial Serives, a national home loan lender and servicer, integrates Google's Gemini LLMs across various departments to enhance efficiency and reduce costs while ensuring security and compliance. HR uses it for job descriptions and policy drafting, while the underwriting team uses Gemini to analyze proprietary data, improving regulatory understanding and best practices. | [🔗](https://workspace.google.com/blog/customer-stories/how-customers-use-gemini-google-workspace-focus-what-they-do-best) | - |
| BNY Mellon - Eliza | Enterprise-Wide | Aug 2024 | BNY Mellon's AI chatbot Eliza leverages multiple LLMs, including OpenAI GPT-4, Google Gemini, and LLaMA, to assist employees with complex queries. It retrieves information from internal databases, streamlining workflows and improving efficiency. Eliza also enables employees to develop AI-driven tools for banking tasks like lead generation | [🔗](https://finance.yahoo.com/news/exclusive-bny-ai-tool-eliza-095400656.html) | - |
| BNP Paribas & Mistral AI | Enterprise-Wide | Jul 2024 | BNP Paribas has partnered with Mistral AI to integrate LLMs across various sectors, including customer support, sales, and IT. This collaboration enables the bank to deploy advanced AI models on-premises, ensuring compliance with regulatory standards. | [🔗](https://group.bnpparibas/en/press-release/bnp-paribas-and-mistral-ai-sign-a-partnership-agreement-covering-all-mistral-ai-models) | - |
| Bitext Mistral-7b-Banking | Specialized Model | Jun 2024 | Fine-tuned version of the Mistral-7B-Instruct-v0.2, specifically tailored for the banking domain. It is optimized to answer questions and assist users with various banking transactions | [🔗](https://github.com/bitext/bitext-mistral-7b-banking) | [🔗](https://www.bitext.com/blog/general-purpose-models-verticalized-enterprise-genai/) |
| Scotiabank & Google | Plug-and-Play | May 2024 | Scotiabank uses LLMs via Google Cloud to enhance customer service and automate document processing. It features LLM-powered chatbot summarization for seamless handoffs and faster resolutions, plus enhanced Q&A and search for streamlined employee workflows. | [🔗](https://www.scotiabank.com/corporate/en/home/media-centre/media-centre/news-release.html) | - |
| ING Bank | Plug-and-Play | Feb 2024 | ING implemented a generative AI-powered customer-facing chatbot to enhance customer service efficiency. This chatbot utilizes a multi-step process that retrieves information from data stores, ranks potential answers by relevance, and applies strict guardrails to ensure accurate and appropriate responses. | [🔗](https://www.mckinsey.com/industries/financial-services/how-we-help-clients/banking-on-innovation-how-ing-uses-generative-ai-to-put-people-first) | - |
| Citi & GitHub Copilot | Plug-and-Play | Feb 2024 |  Citi has focused on improving developer productivity by embracing LLM-based coding assistants by rolling out GitHub Copilot (powered by OpenAI’s Codex) to all of its 40,000 software developers enterprise-wide​. It serves as an AI pair-programmer, suggesting code snippets, functions, or fixes inside developers’ code editors. | [🔗](https://archive.is/C1FLj) | - |
| Akbank & Azure OpenAI | Plug-and-Play | Jan 2024 | Akbank, one of Türkiye's largest banks, uses Azure OpenAI Service to power an AI chatbot that automates customer support, improving accuracy to 90% and cutting response times by three minutes per interaction. This enhances efficiency and allows agents to focus on proactive service. | [🔗](https://www.microsoft.com/en/customers/story/1731060947842679292-akbank-azure-open-ai-service-banking-en-turkiye) | - |
| Ally Financial - Ally.ai | Plug-and-Play | Dec 2023 | Ally Financial, the largest digital-only bank in the US and a leading auto lender, launched Azure OpenAI LLM-powered Ally.ai to more than 700 customer care associates in summarizing conversations between them and Ally customers. This automation of post-call documentation for customer service associates is done through LLM summarization of customer call transcripts. | [🔗](https://www.microsoft.com/en/customers/story/1715820133841482699-ally-azure-banking-en-united-states) | - |
| Westpac & KAI-GPT | Specialized Model | Jun 2023 | Australian lender Westpac is using KAI-GPT, a banking industry-specific LLM, to help bankers locate, interpret and understand information from policies, regulatory filings, procedures, and complex financial products. KAI-GPT is based on Pythia-Chat-Base-7B, fine-tuned on banking-related dataset comprising 24k question-answer pairs from Common Crawl, 18k questions from Kasisto's own conversational data, and 245 million words from 44k banking-related documents  | [🔗](https://www.retailbankerinternational.com/news/digital-experience-platform-kasisto-launches-kai-gpt/?cf-view) | [🔗](https://kasisto.com/blog/kai-gpt-the-first-large-language-model-purpose-built-for-banking/?utm_source=chatgpt.com) |
| XuanYuan 2.0 | Specialized Model | May 2023 | Chat model (built upon the BLOOM-176B architecture) trained by combining general-domain with domain-specific knowledge and integrating the stages of pre-training and fine-tuning, It is capable of providing accurate and contextually appropriate responses in the Chinese financial domain. | - | [🔗](https://arxiv.org/abs/2305.12002) |
| BBT-FinT5 | Specialized Model | Feb 2023 | Chinese financial pre-training language model (1B parameters) based on the T5 model, and pre-trained on the 300Gb financial corpus called FinCorpus | - | [🔗](https://arxiv.org/pdf/2302.09432) |

___
<a name="wealth"></a>
## Wealth Management & Capital Markets
| Name | Type | Date | Description | Site | Paper |
| --- | --- | --- | --- | --- | --- |
| FinBLOOM | Specialized Model | Feb 2025 | FinBloom 7B, built on Bloom 7B, was trained on 14M financial news articles and 12M SEC filings, then fine-tuned with 50K financial queries for enhanced real-time data retrieval. This process ensures strong contextual understanding for financial decision-making. | --- | [🔗](https://arxiv.org/abs/2502.18471) |
| FinE5 | Specialized Model | Feb 2025 | Fin-E5, the finance-adapted embedding model in FinMTEB, is built on the E5 model and trained using a persona-based data synthesis method to enhance performance across financial embedding tasks. | --- | [🔗](https://arxiv.org/abs/2502.10990) |
| MUFG Bank | Plug-and-Play | Feb 2025 | MUFG Bank leveraged LLMs to automate data extraction and summarization from corporate reports, enabling faster financial analysis for FX & Derivative Sales. The system, leveraging retrieval-augmented generation (RAG) and fine-tuned prompts, reduces client presentation preparation from hours to minutes, thereby improving client advisory efficiency. | [🔗](https://blog.langchain.dev/customers-mufgbank/) | - |
| PIMCO & Azure | Enterprise-Wide | Feb 2025 | PIMCO developed ChatGWM, an AI-powered search tool built on Azure AI, to enhance client service by streamlining information retrieval for its associates. ChatGWM utilizes retrieval-augmented generation (RAG) to search across approved structured and unstructured data sources, and then process them with Azure OpenAI LLM to provide accurate and up-to-date information swiftly. | [🔗](https://www.microsoft.com/en/customers/story/19744-pimco-sharepoint) | - |
| Rogo & OpenAI | Specialized Model | Feb 2025 | Rogo's fine-tuned OpenAI models and integration of extensive financial datasets (including S&P Global, Crunchbase, and FactSet) allows it to scale financial analysis and deliver real-time financial intelligence to >5k financial professionals, shifting their focus from manual work to high-value decision making. It uses GPT-4o for in-depth financial analysis, o1-mini for contextualizing financial data, and o1 for evaluations and advanced reasoning. | [🔗](https://openai.com/index/rogo/) | - |
| TigerGPT | Plug-and-Play | Feb 2025 | Tiger Brokers integrated DeepSeek's AI model, DeepSeek-R1, into its AI-powered chatbot, TigerGPT. This adoption aims to enhance market analysis and trading capabilities for its customers through the improved logical reasoning capabilities. | [🔗](https://www.reuters.com/technology/artificial-intelligence/tiger-brokers-adopts-deepseek-model-chinese-brokerages-funds-rush-embrace-ai-2025-02-18) | - |
| Aditya Birla Capital | Plug-and-Play | Jan 2025 | Financial services provider Aditya Birla Capital implemented SimpliFi, a generative AI chatbot built on Azure OpenAI Service, to enhance customer engagement and streamline financial services. SimpliFi assists users in navigating financial solutions independently, aligning with the preferences of their 25 to 35-year-old demographic. | [🔗](https://www.microsoft.com/en/customers/story/20596-aditya-birla-financial-shared-services-azure-open-ai-service) | - |
| Touchstone-GPT | Specialized Model | Nov 2024 | Open-source financial LLM trained through continual pre-training and financial instruction tuning, which demonstrates strong performance on the financial bilingual (English and Chinese) Golden Touchstone benchmark. | [🔗](https://github.com/IDEA-FinAI/Golden-Touchstone) | [🔗](https://arxiv.org/abs/2411.06272) |
| Banca Investis | Plug-and-Play | Nov 2024 | Banca Investis launched NIWA, a GenAI-powered investment advisory platform designed to enhance customer engagement through hyper-personalized financial services. Serving as a "digital junior banker," NIWA analyzes over 500 pieces of information daily—including clients' financial assets, preferences, and market research—to provide tailored financial insights and real-time responses to investment-related inquiries. | [🔗](https://www.bain.com/about/media-center/press-releases/2024/banca-investis-partners-with-bain--company-to-create-a-market-first-investment-advisory-platform-powered-by-generative-ai/) | - |
| FinTral | Specialized Model | Aug 2024 | Suite of multimodal LLMs built upon the Mistral-7b model and tailored for financial analysis. FinTral integrates textual, numerical,  tabular, and image data, and is pretrained on a 20 billion token, high quality dataset | - | [🔗](https://aclanthology.org/2024.findings-acl.774/) |
| Goldman Sachs & Meta | Enterprise-Wide | Aug 2024 | Goldman Sachs introduced the GS AI Platform, a GenAI tool designed to enhance employee productivity across various use cases. Built on Meta's LLaMa models, the assistant aids in tasks such as summarizing and proofreading emails, extracting information from documents, as well as translating code between programming languages. | [🔗](https://www.cnbc.com/2025/01/21/goldman-sachs-launches-ai-assistant.html) | - |
| Nomura & Meta | Enterprise-Wide | Aug 2024 | Leading Japanese financial institution Nomura uses Meta's LLaMa models on Amazon Bedrock to democratize generative AI by driving faster innovation, transparency, bias guardrails, and robust performance across text summarization, code generation, log analysis, and document processing. | [🔗](https://aws.amazon.com/solutions/case-studies/nomura-video-case-study/) | - |
| JPMorgan Chase IndexGPT | Enterprise-Wide | Jul 2024 | JPMorgan Chase launched a generative AI-based tool (via AWS Bedrock) called IndexGPT, designed to serve as a 'research analyst' for over 50,000 employees, aiding in various tasks that enhance productivity and decision-making within the firm. It is able to generate and refine written documents, provide creative solutions and summarize extensive documents. | [🔗](https://qz.com/jpmorgan-indexgpt-ai-chatbot-investment-advice-1850478529) | - |
| IDEA-FinQA | Specialized Model | Jun 2024 | Financial question-answering system based on Qwen1.5-14B-Chat, utilizing real-time knowledge injection and supporting various data collection and querying methodologies, and comprises three main modules: the data collector, the data querying module, and LLM-based agents tasked with specific functions. | [🔗](https://github.com/IDEA-FinAI/IDEAFinBench) | [🔗](https://arxiv.org/abs/2407.00365) |
| Ask FT | Plug-and-Play | Mar 2024 | LLM tool by Financial Times (FT) that enables subscribers to query and receive responses derived from two decades of published FT content. | [🔗](https://aboutus.ft.com/press_release/financial-times-launches-first-generative-ai-tool) | - |
| BCI & Azure | Enterprise-Wide | Mar 2024 | British Columbia Investment Management Corporation (BCI) integrated Microsoft 365 Copilot and Azure OpenAI Service to enhance productivity and streamline operations. This implementation has led to a 10%-20% productivity boost for 84% of initial users, saving over 2,300 person-hours through automation. Notably, the time required to write internal audit reports decreased by 30%. | [🔗](https://www.microsoft.com/en/customers/story/18816-british-columbia-investment-management-corporation-microsoft-365-copilot) | - |
| RAVEN | Specialized Model | Jan 2024 | Fine-tuned LLaMA-2 13B Chat model designed to enhance financial data analysis by integrating external tools. Used supervised fine-tuning with parameter-efficient techniques, utilizing a diverse set of financial question-answering datasets, including TAT-QA, Financial PhraseBank, WikiSQL, and OTT-QA | - | [🔗](https://arxiv.org/abs/2401.15328) |
| InvestLM | Specialized Model | Sep 2023 | Financial domain LLM tuned on LLaMA-65B, using a carefully curated instruction dataset related to financial investment. The small yet diverse instruction dataset covers a wide range of financial related topics, from Chartered Financial Analyst (CFA) exam questions to SEC filings to Stackexchange quantitative finance discussions. | [🔗](https://github.com/AbaciNLP/InvestLM) | [🔗](https://arxiv.org/abs/2309.13064) |
| CFGPT | Specialized Model | Sep 2023 | Financial LLM based on InternLM-7B that is designed to handle financial texts effectively. It was pre-trained on 584 million documents (141 billion tokens) from Chinese financial sources like announcements, research reports, social media content, and financial news, and then fine-tuned on 1.5 million instruction pairs (1.5 billion tokens) tailored for specific tasks of financial analysis and decision-making. | [🔗](https://github.com/TongjiFinLab/CFGPT) | [🔗](https://arxiv.org/abs/2309.10654)
| FinGPT | Specialized Model | Jun 2023 | Open-source financial LLM (FinLLM) using a data-centric approach (based on Llama 2) for automated data curation and efficient adaptation, aiming to democratize AI in finance with applications in robo-advising, algorithmic trading, and low-code development. | [🔗](https://github.com/AI4Finance-Foundation/FinGPT) | [🔗](https://arxiv.org/abs/2306.06031) |
| FinMA | Specialized Model | Jun 2023 | Comprehensive framework that introduces FinMA (Financial Multi-task Assistant), an open-source financial LLM fine-tuned (7B and 30B versions) from LLaMA using a diverse, multi-task instruction dataset of 136,000 samples. The dataset encompasses various financial tasks, document types, and data modalities. | [🔗](https://github.com/chancefocus/PIXIU) | [🔗](https://arxiv.org/abs/2306.05443) |
| Fin-Llama | Specialized Model | Jun 2023 | Specialized version of LLaMA 33B, fine-tuned (with QLoRA and 4-bit quantization) for financial applications using a 16.9k instruction dataset. | [🔗](https://github.com/Bavest/fin-llama) | - |
| Morningstar - Mo chatbot | Plug-and-Play | May 2023 | Morningstar introduced Mo, an AI chatbot powered by the Morningstar Intelligence Engine, which combines Morningstar's extensive investment research library with Microsoft's Azure OpenAI Service. Mo is designed to provide investors and financial professionals with concise, conversational insights by processing natural language queries and summarizing relevant information from over 750,000 investment options. | [🔗](https://newsroom.morningstar.com/newsroom/news-archive/press-release-details/2023/Mo-an-AI-Chatbot-Powered-by-Morningstar-Intelligence-Engine-Debuts-in-Morningstar-Platforms/default.aspx) | - |
| Cornucopia-LLaMA-Fin-Chinese | Specialized Model | Apr 2023 | Open-source LLaMA-based model fine-tuned for Chinese financial applications. It uses instruction tuning with Chinese financial Q&A datasets to enhance domain-specific performance. | [🔗](https://github.com/jerry1993-tech/Cornucopia-LLaMA-Fin-Chinese) | - |
| BloombergGPT | Specialized Model | Mar 2023 | 50-billion-parameter LLM specifically designed for financial applications and the industry's unique terminology, trained on a 363-billion-token dataset sourced from Bloomberg’s proprietary data, complemented with 345 billion tokens from general-purpose datasets | [🔗](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) | [🔗](https://arxiv.org/abs/2303.17564) |
| Morgan Stanley & OpenAI | Pre-trained | Mar 2023 | Morgan Stanley Wealth Management announced a partnership with OpenAI to develop an internal-facing GPT-powered assistant (AI @ Morgan Stanley Assistant), allowing financial advisors to query the bank’s vast research repository and internal knowledge base in natural language | [🔗](https://www.morganstanley.com/press-releases/key-milestone-in-innovation-journey-with-openai) | [🔗](https://openai.com/index/morgan-stanley/) |
| FLANG-ELECTRA | Specialized Model | Oct 2022 | Domain specific Financial LANGuage model (FLANG) which uses financial keywords and phrases for better masking, and built on the ELECTRA-base architecture. *Note: Considered a smaller LM as it has fewer than 1B params* | [🔗](https://github.com/SALT-NLP/FLANG) | [🔗](https://arxiv.org/abs/2211.00083)
| FinBERT-21 | Specialized Model | Jul 2020 | FinBERT (BERT for Financial Text Mining) is a domain specific language model pre-trained on large-scale financial corpora, allowing it to capture language knowledge and semantic information from the finance domain. *Note: Considered a smaller LM as it has fewer than 1B params* | - | [🔗](https://www.ijcai.org/proceedings/2020/622) |

___
<a name="payments"></a>
## Payments & FinTech
| Name | Type | Date | Description | Site | Paper |
| --- | --- | --- | --- | --- | --- |
| FinQuery & Google | Plug-and-Play | Jun 2024 | FinQuery, a fintech company, is using Gemini as a productivity and collaboration tool to help in brainstorming sessions, draft emails 20% faster, manage complex cross-organizational project plans, and aid engineering teams with debugging code and evaluating new monitoring tools. | [🔗](https://workspace.google.com/blog/customer-stories/finquery-innovates-gemini-google-workspace) | - |
| Discover Financial Services & Google | Enterprise-Wide | Apr 2024 | Discover Financial partnered with Google Cloud to utilize LLMs to helps its 10,000 contact center representatives to search and synthesize information across detailed policies and procedures during calls. | [🔗](https://investorrelations.discover.com/newsroom/press-releases/press-release-details/2024/Discover-Financial-Services-Deploys-Google-Clouds-Generative-AI-to-Transform-Customer-Service/default.aspx) | - |
| Adyen | Plug-and-Play | Nov 2023 | Adyen, a publicly-traded financial technology platform, uses LLMs to improve support operations, and enhance efficiency and response times, with smart ticket routing (which assigns tickets based on sentiment and content) and support agent copilot (which help agents answer tickets faster and more accurately) | [🔗](https://blog.langchain.dev/llms-accelerate-adyens-support-team-through-smart-ticket-routing-and-support-agent-copilot/) | - |
| Stripe & OpenAI | Plug-and-Play | Mar 2023 | Stripe integrates OpenAI’s GPT-4 to enhance its payment platform by analyzing business websites for better support, improving developer assistance through technical documentation processing, and detecting fraud by analyzing community interactions. | [🔗](https://openai.com/index/stripe/) | - |


___
<a name="insurance"></a>
## Insurance & Risk Management
| Name | Type | Date | Description | Site | Paper |
| --- | --- | --- | --- | --- | --- |
| Allianz Insurance Copilot | Plug-and-Play | Feb 2025 | An internal generative AI system to assist with claims management. Launched in 2024 for auto claims, it leverages cutting-edge LLMs to streamline workflows and automate key tasks. | [🔗](https://www.allianz.com/en/mediacenter/news/articles/250205-smarter-claims-management-smoother-settlements.html) | - |
| AllianzGPT | Enterprise-Wide | Feb 2025 | Internal chatbot that leverages capabilities from different LLM providers (e.g., OpenAI, DeepSeek) on Azure to support general productivity tasks, as well as specific functional areas like audit, Risk Consulting actuarial and other business areas. | [🔗](https://www.allianz.com/en/mediacenter/news/articles/250218-ai-at-allianz-the-impact-of-allianzgpt.html) | - |
| Newfront & Anthropic | Plug-and-Play | Dec 2024 | Newfront, an insurance platform serving 20% of U.S. startups with unicorn status, integrated Anthropic's Claude AI to enhance efficiency and client service. Claude automates complex insurance tasks, such as answering detailed benefits questions, reviewing contracts, and processing loss run documents. | [🔗](https://www.anthropic.com/customers/newfront) | - |
| Open-Insurance-LLM-Llama3-8B | Specialized Model | Nov 2024 | Llama 3 (8B model) fine-tuned with LoRA on the InsuranceQA dataset – a corpus of insurance domain Q&A pairs – to specialize it for insurance queries and conversations | [🔗](https://huggingface.co/Raj-Maharajwala/Open-Insurance-LLM-Llama3-8B/tree/main) | - |
| Zurich Insurance & OpenAI | Plug-and-Play | Nov 2024 | Zurich Insurance Group turned to Microsoft Azure OpenAI Service to develop AI applications that lead to more accurate and efficient risk management evaluations, accelerating the underwriting process, reducing turnaround times, and increasing customer satisfaction. | [🔗](https://www.microsoft.com/en/customers/story/19760-zurich-insurance-azure-open-ai-service) | - |
| Zurich Insurance & Azure | Plug-and-Play | Nov 2024 | Zurich Insurance Group leverages the LLM capabilities in Azure OpenAI Service to enhance its underwriting process by converting unstructured customer data—such as images, emails, and reports in various languages—into structured, actionable insights. | [🔗](https://www.microsoft.com/en/customers/story/19760-zurich-insurance-azure-open-ai-service) | - |
| EXL Insurance LLM | Specialized Model | Sep 2024 | Industry-specific LLM that supports critical claims and underwriting-related tasks, such as claims reconciliation, data extraction and interpretation, question-answering, anomaly detection and chronology summarization. EXL utilized NVIDIA NeMo end-to-end platform for the fine-tuning process on 2 billion tokens of private insurance data. | [🔗](https://www.exlservice.com/about/newsroom/exl-launches-specialized-insurance-large-language-model-leveraging-nvidia-ai-enterprise) | - |
| Swiss Re & mea | Plug-and-Play | Sep 2024 | Swiss Re partnered with mea (a generative AI-powered global platform) for insurance process automation, such as extracting unstructured data from submission-related documents (e.g., Schedules of Value) and convert it into structured, analysable data. | [🔗](https://www.reinsurancene.ws/swiss-re-selects-mea-platforms-genai-solution-to-enhance-operations/) | - |
| Bitext Mistral-7B-Insurance | Specialized Model | Jul 2024 | Fine-tuned version of Mistral-7B-Instruct-v0.2, specifically tailored for the insurance domain. It is optimized to answer questions and assist users with various insurance-related procedures | [🔗](https://huggingface.co/bitext/Mistral-7B-Insurance) | [🔗](https://www.bitext.com/blog/general-purpose-models-verticalized-enterprise-genai/) |
| Five Sigma - Clive | Plug-and-Play | Jul 2024 | Insurance tech startup Five Sigma leveraged Google's Gemini models to create an AI claims adjuster engine (Clive) which frees up human claims handlers to focus on areas where a human touch is valuable, like complex decision-making and empathic customer service. | [🔗](https://services.google.com/fh/files/misc/fivesigma_whitepaper.pdf) | - |
| New York Life | Enterprise-Wide | Jul 2024 | New York Life, the largest mutual life insurance company in the US, is actively integrating GenAI across various business functions to enhance efficiency e.g., underwriting, customer service, and hiring. | [🔗](https://www.cxotalk.com/episode/generative-ai-and-business-transformation-at-new-york-life) | - |
| Trumble Insurance Agency & Google | Plug-and-Play | May 2024 | Trumble Insurance Agency is using Gemini for Google Workspace to significantly improve its creativity and the value that it delivers to its clients with enhanced efficiency, productivity, and creativity. | [🔗](https://www.youtube.com/watch?v=V2gwtZJsKqw) | - |
| AXA Secure GPT | Enterprise-Wide | Apr 2024 | AXA developed AXA Secure GPT, a generative AI platform powered by Azure OpenAI Services, to enhance employee productivity while ensuring data security. This platform enables AXA's 140k employees to utilize AI tools within a secure environment, facilitating tasks such as drafting reports, summarizing documents, and generating content. | [🔗](https://www.microsoft.com/en/customers/story/1760377839901581759-axa-gie-azure-insurance-en-france) | - |
| Allstate Insurance | Plug-and-Play | Mar 2024 | Allstate implemented generative AI models, specifically OpenAI's GPT, customized with company-specific terminology, to enhance customer communications and experience. This system generates approximately 50k daily emails for claims representatives, ensuring messages are clear, empathetic, and free from industry jargon. | [🔗](https://www.bcg.com/capabilities/artificial-intelligence/client-success/improving-customer-experiences-with-gen-ai-tools) | - |
| Groupama & Azure OpenAI | Plug-and-Play | Feb 2024 | Groupama, a leading French mutual insurance group, uses Azure OpenAI Service to power a virtual assistant that helps customer managers respond to policyholder inquiries with 80% accuracy, improving efficiency and response quality. It is done through providing pre-written responses to policyholder inquiries, drawing from a comprehensive and secure documentation corpus. | [🔗](https://www.microsoft.com/en/customers/story/1741559204804365124-groupama-azure-openai-service-banking-en-france) | - |
| LAQO insurance & Azure OpenAI | Plug-and-Play | Nov 2023 | LAQO, Croatia's first fully digital insurer, partnered with Infobip to develop Pavle, a 24/7 AI assistant powered by Azure OpenAI Service. Pavle resolves 30% of customer queries, allowing human agents to focus on complex cases and customer acquisition. | [🔗](https://www.microsoft.com/en/customers/story/1705644076562905842-laqo-azure-insurance-en-croatia) | - |
| Roots Automation - InsurGPT | Specialized Model | May 2023 | Roots Automation released an LLM fine-tuned on insurance-specific documents (ACORD forms, First Notice of Loss (FNOL), loss runs) to automate and cut claims processing time significantly | [🔗](https://www.prnewswire.com/news-releases/roots-automation-introduces-insurgpt---the-worlds-most-advanced-generative-ai-model-for-insurance-301823620.html) | - |
