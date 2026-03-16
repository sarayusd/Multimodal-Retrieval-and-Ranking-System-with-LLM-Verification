def format_retrieved_context(retrieved_results, top_n=3):

    context_lines = []

    for i, r in enumerate(retrieved_results[:top_n]):

        caption = r["caption"]

        audit_score = r.get("audit_score", None)

        audit_verdict = r.get("audit_verdict", None)

        extra = ""

        if audit_score is not None:
            extra = f" | audit_score={audit_score:.1f}"

        if audit_verdict is not None:
            extra += f" | verdict={audit_verdict}"

        context_lines.append(f"{i+1}. {caption[:220]}{extra}")

    return "\n".join(context_lines)