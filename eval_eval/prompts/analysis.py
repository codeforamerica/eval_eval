import textwrap


def prompt_1(notice: str) -> str:
    return textwrap.dedent(
        f"""
        The following document is a sample California Supplemental Nutrition Assistance Program (SNAP) notice of benefits approval. 

        **Instructions:**
        Write a 2-3 sentence summary of the notice and evaluate the document's quality by answering questions about it in 2-3 sentences. 

        **Questions:**
        - What actions are required by the recipient?
        - Is the document primarily informational or is action required?
        - Is this notice written in plain language, at 6th-grade reading level or lower?
        - How could this document be more effective for the recipient?

        **Document to Analyze:**
        {notice}
    """
    )


def prompt_2(notice: str) -> str:
    return textwrap.dedent(
        f"""
        You are analyzing a California Supplemental Nutrition Assistance Program (SNAP) notice of benefits approval. Your goal is to provide a clear summary and thorough evaluation of the document's effectiveness for the recipient.

        **Document Summary:**
        Write a 2-3 sentence summary that captures the key information and purpose of this notice.

        **Document Analysis:**
        Answer each question below with 2-3 well-reasoned sentences that demonstrate specific analysis of the document:

        - **Required Actions**: What specific actions, if any, must the recipient take after receiving this notice? Include deadlines and consequences of inaction.

        - **Document Classification**: Determine whether this document is primarily informational (notifying the recipient of status or updates) or action-required (demanding a response to maintain benefits). Explain the potential consequences if the recipient does not respond to or act on this document.

        - **Plain Language Assessment**: Evaluate whether this notice uses plain language appropriate for a 6th-grade reading level. Consider vocabulary complexity, sentence structure, and use of jargon or technical terms.

        - **Effectiveness Improvements**: Identify the most significant changes that would make this document more effective for the recipient, focusing on clarity, accessibility, and actionability.

        **Document to Analyze:**
        {notice}
    """
    )
