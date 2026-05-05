from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from src.workflow.chains.answer_grader import answer_grader
from src.workflow.chains.hallucination_grader import hallucination_grader
from src.workflow.chains.router import RouteQuery, question_router
from src.workflow.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from src.workflow.nodes.generate import generate
from src.workflow.nodes.grade_documents import grade_documents
from src.workflow.nodes.retrieve import retrieve
from src.workflow.nodes.web_search import web_search
from src.workflow.state import GraphState

load_dotenv()

MAX_RETRIES = 3  # Maximum times to loop through generate → grade before giving up


def decide_to_generate(state: GraphState) -> str:
    """Route to web search or generation."""
    print("---ASSESS DOCUMENTS---")
    return WEBSEARCH if state["web_search"] else GENERATE


def increment_retry(state: GraphState) -> dict:
    """
    Increment the retry counter each time we loop back into generate.
    This node is inserted before GENERATE so we always have an up-to-date count.
    """
    current = state.get("retry_count", 0)
    return {"retry_count": current + 1}


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")

    if state.get("retry_count", 0) >= MAX_RETRIES:
        print(f"---MAX RETRIES REACHED---")
        return "max_retries_exceeded"

    question   = state["question"]
    documents  = state["documents"]
    generation = state["generation"]

    # FIX 1: extract plain text from Document objects
    if documents:
        docs_text = "\n\n".join(
            d.page_content if hasattr(d, "page_content") else str(d)
            for d in documents
        )
    else:
        # FIX 2: no documents — skip hallucination check entirely
        print("---NO DOCS — skipping to answer grader---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        return "useful" if score.binary_score else "not useful"

    score = hallucination_grader.invoke({
        "documents": docs_text,
        "generation": generation,
    })

    # FIX 3: log what the grader actually returned so you can debug
    print(f"---HALLUCINATION SCORE: {score.binary_score}---")

    if score.binary_score:
        score = answer_grader.invoke({"question": question, "generation": generation})
        print(f"---ANSWER SCORE: {score.binary_score}---")
        return "useful" if score.binary_score else "not useful"
    else:
        return "not supported"



def route_question(state: GraphState) -> str:
    """Route question to vectorstore or web search."""
    print("---ROUTE QUESTION---")
    source: RouteQuery = question_router.invoke({"question": state["question"]})
    return WEBSEARCH if source.datasource == WEBSEARCH else RETRIEVE


# ─── Build workflow ────────────────────────────────────────────────────────────

workflow = StateGraph(GraphState)

# Register nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

# NEW: a lightweight node that just bumps retry_count before every generation
workflow.add_node("increment_retry", increment_retry)

# Entry point: route to vectorstore retrieval or straight to web search
workflow.set_conditional_entry_point(
    route_question,
    {WEBSEARCH: WEBSEARCH, RETRIEVE: RETRIEVE},
)

# After retrieval, grade the documents
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

# After grading, either go to web search (irrelevant docs) or generate
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH, GENERATE: "increment_retry"},  # bump counter first
)

# Always increment before generating
workflow.add_edge("increment_retry", GENERATE)

# After generation, grade for hallucinations and usefulness
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": "increment_retry",   # ← was GENERATE, now increment first
        "useful": END,
        "not useful": WEBSEARCH,              # WEBSEARCH already → increment_retry → GENERATE
        "max_retries_exceeded": END,
    },
)

# Web search always feeds back into generation
workflow.add_edge(WEBSEARCH, "increment_retry")

app = workflow.compile()