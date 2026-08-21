"""
Title: Basic Chain (Google Generative AI)

uv run python src/01_b_basic_chain_google.py
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    load_dotenv()

    prompt = ChatPromptTemplate.from_template("Translate to Hindi: {text}")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite"
    )

    # The pipe operator composes components
    chain = prompt | llm
    # Note: we kept the raw response (no parser) to inspect metadata

    response = chain.invoke({"text": "Hello!, my name is Abhishek."})
    print(f"Response content: {response.content}")
    print(f"Response Usage Metadata: {response.usage_metadata}")



if __name__ == "__main__":
    main()



"""
-----------------------------------------
Output:
-----------------------------------------


Response content: [{'type': 'text', 'text': 'नमस्ते! मेरा नाम अभिषेक है।\n\n(Namaste! Mera naam Abhishek hai.)', 'extras': {'signature': 'EnEKbwERTTIP9mvAUBI+5pKapjupnjm2JzqrIB0wSwgGGpsUjbF0LGK7zI3XqhSEOITv5ShYV4Fw+7pGeEgd0DJc+21IvE2+5+YODfCWnxgQbW7sgMwoUiAwT884OxG7YIIo3gihlktzaBsErSpEXo3DwA=='}}]
Response Usage Metadata: {'input_tokens': 12, 'output_tokens': 17, 'total_tokens': 29, 'input_token_details': {'cache_read': 0}}


"""