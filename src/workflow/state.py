from typing import List, TypedDict

class GraphState(TypedDict):
    """State object for workflow containing query, documents and control flags"""
    question: str           # User's original query
    generation: str         # LLM generated response
    web_search: bool        # Control flag for web search requirement
    documents: List[str]    # Retrieved documents relevant to the query

# class GraphState(TypedDict):
#     question: str
#     generation: str
#     use_rag: bool
#     use_web: bool
#     vector_docs: List[str]
#     web_docs: List[str]

# class RetrievedDoc(TypedDict):
#     content: str
#     source: str        # "vector", "web"
#     score: float
#     url: str | None
