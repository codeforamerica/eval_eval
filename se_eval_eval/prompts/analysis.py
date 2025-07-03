import textwrap

ANALYSIS_PROMPT = textwrap.dedent("""
    Answer the following questions about the provided Supplemental Nutrition Assistance Program (SNAP) client notice.

    Instructions: 
        - Be sure to answer all 10 questions
        - Restate the question
        - Provide a "Yes", "No" or "IDK" answer
        - Always provide a reason for your answer   
        - Use "IDK" when information might be present but is unclear or ambiguous
        - Use "No" when information is clearly absent
        - For the plain language question, consider the overall readability, not just individual elements

    Questions:
        - Does the notice clearly state what specific action the state plans to take (e.g., benefit approval, reduction, termination, denial)?
        - Does the notice explain the specific factual or legal basis for the proposed action?
        - Does the notice include the household's right to request a fair hearing?
        - Does the notice include a telephone number for the SNAP office that is either toll-free OR accepts collect calls for households outside the local calling area?
        - Does the notice include either a specific contact person's name OR a general contact role/department for additional information?
        - Does the notice include the availability of continued benefits?
        - Does the notice include the liability of the household for any over issuances received while awaiting a fair hearing if the hearing official's decision is adverse to the household?
        - Does the notice inform the household about the availability of free legal representation or advocacy services?
        - Is the notice written in plain language that would be understandable to someone with a 6th grade reading level or below? Consider sentence length, word complexity, jargon usage, and overall clarity.
        - The above questions represent best practices for writing SNAP client notices. Overall, does this notice meet those requirements? 

    Notice:
    {notice}

    IMPORTANT: remember to ANSWER ALL 10 QUESTIONS
""")