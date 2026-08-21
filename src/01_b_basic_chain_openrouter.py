"""
Title: Basic Chain (OpenRouter)

uv run python src/01_b_basic_chain_openrouter.py
"""

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

def main():
    load_dotenv()

    prompt = ChatPromptTemplate.from_template("Translate to Hindi: {text}")
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct"
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

Response content: नमस्ते!, मेरा नाम अभिषेक है।
Response Usage Metadata: {'input_tokens': 42, 'output_tokens': 27, 'total_tokens': 69, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'reasoning': 0}}


"""