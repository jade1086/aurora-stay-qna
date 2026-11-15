
from langchain_openai import ChatOpenAI
import os, re, json, time

# Number of FAISS candidates passed to the LLM re-ranker
FETCH_K = 15

RERANK_PROMPT = """You are the re-ranking step of a retrieval system for AuroraStay Hotels policy documents.

Question:
{query}

Candidate passages:
{passages}

Task: select ONLY the passages that contain information needed to answer some part of the question
(for example, the exact property row, rate rule, fee, time, or procedure that is asked about).
Exclude passages that are only loosely related, generic, or about other topics.
Order the selected passages from most to least relevant and select at most {k}.
If no passage is relevant, select the single most related passage.

Return only JSON in this format: {{"selected": [passage numbers]}}"""


# Re-rank the FAISS candidates with an LLM and keep only the passages needed to answer the query
def rerank(query, candidates, k, model_name="gpt-4o-mini"):
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        max_tokens=100,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_API_BASE")
    )
    passages = "\n\n".join(f"[{i}]\n{doc.page_content}" for i, doc in enumerate(candidates))
    prompt = RERANK_PROMPT.format(query=query, passages=passages, k=k)

    for attempt in range(3):
        try:
            raw = model.invoke(prompt).content
            selected = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["selected"]
            selected = [int(i) for i in selected if 0 <= int(i) < len(candidates)]
            selected = list(dict.fromkeys(selected))[:k]
            return [candidates[i] for i in selected] or candidates[:1]
        except Exception:
            time.sleep(5)

    # Fall back to the dense ranking if the re-ranker is unavailable
    return candidates[:k]


# Two-stage retrieval: FAISS similarity search for recall, LLM re-ranking for precision
def retrieve(query: str, k: int = 2, vectorstore=None) -> list[str]:
    candidates = vectorstore.similarity_search(query, k=max(FETCH_K, k))
    results = rerank(query, candidates, k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant that helps AuroraStay Hotels staff answer guest questions using AuroraStay's hotel policy documents.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Use only information explicitly supported by the retrieved policy information.
    2. Do not use outside knowledge, assumptions, or information not present in the retrieved policy information.
    3. Include all relevant information needed to answer the query completely, including exact times, amounts, limits, and property codes (for example, ASH-CHI Loop) when they are stated.
    4. Do not omit important conditions, exceptions, approvals, or limitations that apply to the answer (for example, "subject to availability", "may incur a fee", "manager approval", or "the confirmation controls").
    5. Focus only on information directly relevant to the user's query.
    6. Do not include unrelated information from the retrieved policy information.
    7. Answer the user's specific question directly in the first sentence, and make sure every part of a multi-part question is answered.
    8. Keep the answer concise, clear, and easy to understand (typically 2 to 5 sentences). Do not add generic closing advice that is not stated in the policy.
    9. Do not invent fees, times, room features, amenities, properties, exceptions, or approval rules, and do not soften or strengthen the policy wording (for example, do not add "typically" or "always" unless the policy says so).
    10. If the information is insufficient, say:
        "The available policy documents do not contain enough information to answer this question."
        If only part of the question can be answered, answer that part and state which part is not covered. If the retrieved information conflicts, clearly mention the conflict.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
