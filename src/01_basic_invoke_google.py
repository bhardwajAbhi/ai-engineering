"""
Title: Basic LLM Invocation (Google Generative AI)
uv run python src/01_basic_invoke_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    load_dotenv()

    # Initialize the model
    # For reproducible demos, keep temperature at 0 and bound the output length.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_output_tokens=128,
    )

    # Factual question
    response = llm.invoke("What is the capital of France?")
    print(f"Response: {response.content}")

    # Inspect metadata (usage tokens)
    print(f"Usage Metadata: {response.usage_metadata}")

    # Example: Controlling output length
    print("\n--- Output Token Limit Example ---")
    short_model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", max_output_tokens=5)
    short_response = short_model.invoke(
        "Tell me a very long story about the history of the universe."
    )
    print(f"Short Response: {short_response.content}...")
    print(f"Finish Reason: {short_response.response_metadata.get('finish_reason')}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


Response: [{'type': 'text', 'text': 'The capital of France is Paris.', 'extras': {'signature': 'EnEKbwERTTIPETeczNYlpG4STNHMNYIxBAlqcPlg54TxLgU4aQUqBdEO0m5iwc7s1F54lV2UJfuDD3od8iDFA2NVMOpsLMkwe+yfeFqYWjDaC5KJaC8qLkk22DMJhyY4JyCMVoXSlGL6VCZHn8MgsIaqRQ=='}}]
Usage Metadata: {'input_tokens': 8, 'output_tokens': 7, 'total_tokens': 15, 'input_token_details': {'cache_read': 0}}

--- Output Token Limit Example ---
Short Response: [{'type': 'text', 'text': 'To', 'extras': {'signature': 'EnEKbwERTTIPCsHJbYOfitoN2A/QQf8DzNCJzyJGJh2/BXEd3iixGSzDBCucVGtrEIworL032Cnt8JrTZDnHupo0MlgMhaFrvVRpFIz8QENepKm8jl+Js9sjPqR3COegImWOeJf2pLCjj295+ASuCTHTMw=='}}]...
Finish Reason: MAX_TOKENS


"""