import textwrap

def baseline_prompt(from_lang, to_lang, text):
    return textwrap.dedent(f"""
            Translate the following text from {from_lang} to {to_lang}.
            
            Text: {text}
            
            Return only valid JSON in this exact format:
            {{"text": "your translation here", "language": "{to_lang}"}}
            
            Example:
            Input: "I love pizza topped with pineapple." (English to French)
            Output: {{"text": "J'aime la pizza garnie d'ananas.", "language": "French"}}
        """
    )
