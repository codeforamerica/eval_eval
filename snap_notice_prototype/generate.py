import glob
import json
from typing import List, Literal
import textwrap

from pydantic import BaseModel, Field
import ollama


GENERATOR_MODEL = "deepseek-r1:8b"


"""
GENERATE ANALYSIS
"""

class AnalysisResult(BaseModel):
    question: str = Field(description="The question being answered")
    answer: Literal["Yes", "No", "IDK"] = Field(description="Your answer")
    reason: str = Field(description="The explanation for your answer")

class Analysis(BaseModel):
    results: List[AnalysisResult]


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

def generate_analysis(text: str) -> Analysis:
    prompt = ANALYSIS_PROMPT.format(notice=text)
    resp = ollama.generate(GENERATOR_MODEL, prompt=prompt, stream=False, format=Analysis.model_json_schema())
    response_model = Analysis.model_validate_json(resp.response)
    if len(response_model.results) != 10:
        raise RuntimeError("Not all questions were answered.")
    return response_model



if __name__ == "__main__":
  #  with open("context_documents/ca_approval.txt") as file:
        # original_response = generate_notice(file.read())
        # print("HERE IS THE ORIGINAL NOTICE:")
        # print(original_response.notice_text)
        # simplified_response = simplify_notice_language(original_response.notice_text)
        # print("HERE IS THE SIMPLIFIED NOTICE:")
        # print(simplified_response.notice_text)

    output = []
    for file_path in glob.glob("context_documents/*.txt"):
        with open(file_path) as file:
            text = file.read()
        print(f"Analyzing {file_path}...")
        response = generate_analysis(text)
        output.append({"file": file_path, "response": response.model_dump()})
    with open("notice_analysis.json", "w") as output_file:
        output_file.write(json.dumps(output, indent=2))