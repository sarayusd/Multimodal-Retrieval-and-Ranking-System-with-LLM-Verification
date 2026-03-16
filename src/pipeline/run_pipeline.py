from retrieval.fallback_retrieval import retrieve_with_fallback
from llm.vision_auditor import visual_auditor
from rag.context_formatter import format_retrieved_context
from rag.explanation_generator import generate_scene_explanation


def run_agent_pipeline(query, retrieval_fn):

    results = retrieval_fn(query)

    audited = []

    for r in results:

        audit = visual_auditor(query, r["image_path"])

        r.update(audit)

        audited.append(r)

    context = format_retrieved_context(audited)

    explanation = generate_scene_explanation(query, context)

    return {
        "results": audited,
        "explanation": explanation
    }