# Multimodal-Scene-Retrieval-System-with-Hybrid-Search-and-RAG

A multimodal retrieval and reasoning system that supports **text→image, image→image, and image→caption search** using vision-language embeddings and large language models. Images and captions from the **COCO captions dataset** are encoded using **OpenCLIP** and indexed in a **ChromaDB vector database** for similarity search.

The system combines **dense CLIP retrieval, BM25 lexical retrieval, SentenceTransformers cross-encoder reranking, GPT-4o Vision-based verification, and few-shot RAG prompting** to improve retrieval robustness and generate grounded scene explanations from retrieved evidence.

## Key Features

* **Multimodal Embedding Space:** Uses **OpenCLIP (ViT-L-14)** to encode images and captions from the **COCO captions dataset** into a shared embedding space.

* **Vector Database Search:** Indexes image and caption embeddings in **ChromaDB** to support persistent multimodal similarity search.

* **Multi-Stage Retrieval Pipeline:** Combines **dense CLIP vector retrieval, BM25 lexical retrieval, and SentenceTransformers cross-encoder reranking** to improve retrieval quality.

* **Fallback Retrieval Workflow:** Implements a retrieval cascade  
  `reranked → hybrid → dense → BM25`  
  to ensure relevant results are returned across different query types.

* **Vision-Based Verification:** Retrieved images are evaluated using **GPT-4o Vision**, which assigns an `audit_score` and `audit_verdict` to assess image relevance before explanation generation.

* **Few-Shot Grounded RAG:** Uses a **LangChain PromptTemplate with few-shot examples** to generate scene explanations using retrieved captions as evidence.

* **Evaluation Framework:** Includes automated evaluation computing **Recall@K, Precision@K, MRR, nDCG@K, and latency** for retrieval performance.

---

## Architecture

![Architecture](images/agent.png)

The system follows a modular pipeline:

1. **Data Ingestion:** Load the **COCO captions dataset** and map captions to image paths.  
2. **Encoding:** Images and captions are encoded using **OpenCLIP** to generate multimodal embeddings.  
3. **Storage:** Image and caption embeddings are indexed in **ChromaDB** for vector similarity search.  
4. **Retrieval:** The system supports **text→image, image→image, and image→caption retrieval** through dense retrieval, followed by **hybrid retrieval and cross-encoder reranking**.  
5. **Verification:** Retrieved images are checked using **GPT-4o Vision auditing** to evaluate query relevance.  
6. **Reasoning:** Retrieved captions are formatted as context and passed into a **few-shot RAG prompt** to generate a grounded scene explanation.

---

## Tech Stack

* **Languages:** Python  
* **Vision-Language Models:** OpenCLIP (ViT-L-14)  
* **Vector Database:** ChromaDB  
* **Retrieval and Ranking:** BM25, SentenceTransformers cross-encoder  
* **LLM Reasoning:** LangChain, OpenAI API (**GPT-4o Vision**, **GPT-4o-mini**)  
* **Data Processing:** NumPy, Pandas  
* **Visualization:** Matplotlib
