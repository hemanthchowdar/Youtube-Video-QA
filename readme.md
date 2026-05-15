### Problem Statement
Build a RAG based system to chat in real time with youtube video.

**Query's :**

1. Does this video explain about ai.
2. Summarize video.
3. Ask any doubt from the video.

**Flow :**
1. Loading transcript from youtube video.
2. Chunking
3. Embedding
4. Storing in Vector DB.
5. Create a Retriver
6. Bulding Prompt
7. prompt -> LLM -> response
8. Streamlit interface


### Improvements

1. UI based enhancements.
2. Evaluation 
    a. Rages
    b. LangSmith
3. Indexing
    a. Document Injestion
    b. Text Splitting
    c. Vector Space
4. Retrieval
    a. Pre-Retrieval
        * Query rewriting using LLM
        * Multi-query generation
        * Domain aware routing
    b. During Retrieval
        * MMR
        * Hybrid Retrieval
        * Reranking
    c. Post-Retrieval
        * contextual compression
5. Augmentation
    * Prompt templating
    * Answer Grounding
    * Context window optimization

6. Generation
    * Answer with citation.
    * Gaurd railing.

7. System Design
    * multimodal
    * Agentic
    * Memory based