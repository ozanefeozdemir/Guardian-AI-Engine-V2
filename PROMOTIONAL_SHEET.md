# An Innovative Graduation Project from Başkent University: AI-Powered Intrusion Detection System "Guardian"

Başkent University Computer Engineering students Ozan Efe Özdemir and Alperen Melih Göncü developed a notable graduation project under the supervision of Dr. Didem Ölçer during the 2025-2026 academic year. The system they built, named "Guardian AI Engine V2", brings modern deep learning to the heart of network security and detects cyber attacks on live network traffic in real time.

## Deep Learning at the Core of Network Defense

Guardian's most distinctive aspect is its ability to inspect network flows the moment they are observed and classify them as benign or malicious using deep neural networks. Rather than relying on signature databases or shallow classifiers, the project is built around two complementary deep-learning models:

- **FlowGuard (Transformer-based detector):** A Transformer encoder operating on 53 NetFlow-v3 features, hardened through domain-adversarial training across four major intrusion-detection datasets (UNSW-NB15, BoT-IoT, ToN-IoT and CIC-IDS). The added domain discriminator forces the model to learn dataset-invariant attack patterns, dramatically improving generalization to unseen networks.
- **GuardianHybrid (Conv1D + LSTM):** A hybrid architecture that combines 1-D convolutional feature extraction with an LSTM that consumes a 10-step sliding window of recent flows. This temporal view enables fine-grained, multi-class detection across five categories: Benign, DDoS, PortScan, WebAttack and Botnet.

## A Pluggable Model Architecture

A core engineering achievement of the project is its **model provider abstraction**. Through a single `get_model_provider(name)` call, the analysis engine can transparently switch between models — FlowGuard, GuardianHybrid or a legacy scikit-learn baseline — without any change to the surrounding code. This design turns Guardian into a research-friendly platform where new detection models can be integrated, benchmarked and deployed with minimal friction.

## A Different Experience in Every Network

Guardian operates in two modes: a **simulation mode** that replays CSV-based traffic captures for evaluation and demonstration, and a **live mode** that captures packets directly from the network interface. Packets are grouped into bidirectional flows through dedicated flow trackers — including an nDPI-aware extractor that produces NetFlow-v3 features compatible with FlowGuard — and inference runs in milliseconds. As a result, every network environment receives a tailored, real-time defense.

## A Modern Full-Stack Architecture

The project stands out not only for its ML core but also for the engineering depth of its full-stack architecture. A **FastAPI** backend, **Redis**-based alert queue, **PostgreSQL** persistence layer and a **React + Vite** frontend communicate through **WebSockets** to deliver alerts to the security analyst's screen the moment they occur. JWT-based authentication, an audit log of login activity and a Docker-Compose orchestration of the full stack make the system both secure and easy to deploy.

The frontend offers two dashboard experiences — a Classic v1.0 view and a Modern v2.0 panel with dedicated components such as IP rule management — letting analysts choose the workflow that best fits their style while watching live attack alerts stream in.

## A Technically Distinguished Achievement

At the end of two semesters of development, the team successfully integrated Transformer and recurrent deep-learning models, distributed systems and modern web technologies into a single coherent product. Guardian is a forward-looking example of how artificial intelligence can be embedded into critical cybersecurity infrastructure — and a proof that student-built systems can reach production-grade quality.

Guardian aspires to be one of the pioneering projects in AI-driven cybersecurity developed in Türkiye. The natural fit between network flow data and modern deep-learning methods makes the **integration of artificial intelligence into security operations** both feasible and powerful, and it establishes a strong foundation for future innovation in this field. In this respect, Guardian is poised to inspire upcoming projects at the intersection of cybersecurity and artificial intelligence.
