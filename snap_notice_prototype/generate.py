import json
from typing import List
import textwrap

from pydantic import BaseModel, Field
import ollama


GENERATOR_MODEL = "deepseek-r1:8b"

class Case(BaseModel):
    first_name: str
    last_name: str
    case_number: str


class CaseList(BaseModel):
    cases: List[Case]


CLIENT_PROMPT = textwrap.dedent("""
    Generate {count} synthetic legal cases. Return them as a JSON object with a "cases" array containing {count} case objects.

    Each case should have realistic first_name, last_name, and case_number values.
    
    Example:

    [{{
      "first_name": "Alfred",
      "last_name": "Jenkins",
      "case_number": 4277799911
    }},
    {{
      "first_name": "Hop",
      "last_name": "Roberts",
      "case_number": 2331231231
    }},
    {{
      "first_name": "Fulton",
      "last_name": "Hyatt",
      "case_number": 4441234561
    }}]
""")


def generate_clients(n: int) -> CaseList:
    print(f"Generating {n} cases...")
    prompt = CLIENT_PROMPT.format(count=n)
    ret = ollama.generate(GENERATOR_MODEL, prompt=prompt, format=CaseList.model_json_schema(), stream=False)
    list = CaseList.model_validate_json(ret.response)
    assert len(list.cases) == n
    return list


class GeneratedNotice(BaseModel):
    notice_text: str = Field(description="The SNAP notice text.")
    thinking: str = Field(description="Notes on your thinking process.")

NOTICE_PROMPT = textwrap.dedent("""
    You are generating a real, complete Supplemental Nutrition Assistance Program (SNAP) notice for the state of California. This must be a fully filled-out notice with NO placeholders, brackets, or template markers.
    
    CRITICAL INSTRUCTIONS:
    - Replace ALL placeholders with actual information
    - Use the provided case information to fill in specific details
    - Generate realistic details for any missing information (addresses, phone numbers, dates, etc.)
    - Write as if this is an actual government notice being sent to a real person
    
    Required Elements (must all be included with specific details):
        - The proposed action;
        - The reason for the proposed action;
        - The household's right to request a fair hearing;
        - The telephone number of the SNAP office (toll-free number or a number where collect calls will be accepted for households outside the local calling area);
        - If possible, the name of the person to contact for additional information;
        - The availability of continued benefits;
        - The liability of the household for any over issuances received while awaiting a fair hearing if the hearing official's decision is adverse to the household; and
        - If there is an individual or organization available that provides free legal representation information for the household on the availability of the service.
    
    Use this case information to personalize the notice:
    - Recipient Name: {first_name} {last_name}
    - Case Number: {case_number}
    - Generate realistic details for: current date, deadline dates, office address, contact information
    
    Example Notice Structure:
    {example}
    
    Generate a complete, professional government notice. Every detail must be filled in with realistic information. Think of this as a real notice that would be mailed to someone's home.
    
    Return your response as a JSON object with "notice_text" containing the complete notice and "thinking" explaining your approach.
""")


def generate_notices(case_list: CaseList) -> List[str]:
    print("Generating notices now...")
    with open("context_documents/snap_verification_notice_untokenized.txt") as file:
        example_text = file.read()

    notices = []
    for case in case_list.cases:
        print(f"Generating notice for {case.first_name} {case.last_name}...")
        prompt = NOTICE_PROMPT.format(**{**dict(case), **{"example": example_text}})
        resp = ollama.generate(GENERATOR_MODEL, prompt=prompt, format=GeneratedNotice.model_json_schema(), stream=False)
        notice_object = GeneratedNotice.model_validate_json(resp.response)
        print(notice_object)
        notices.append(notice_object.notice_text)
    return notices


if __name__ == "__main__":
    clients = generate_clients(5)
    notices = generate_notices(clients)
    with open("output.json", "w") as file:
        file.write(json.dumps(notices))