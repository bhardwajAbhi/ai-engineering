"""
Title: Basic LLM Invocation (OpenRouter)
uv run python src/01_basic_invoke_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


def main():
    load_dotenv()

    # Initialize the model
    # For reproducible demos, keep temperature at 0 and bound the output length.
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
        max_tokens=128,
    )

    # Factual question
    response = llm.invoke("What is the capital of France?")
    print(f"Response: {response.content}")

    # Inspect metadata (usage tokens)
    print(f"Usage Metadata: {response.usage_metadata}")

    # Example: Controlling output length
    print("\n--- Output Token Limit Example ---")
    short_model = ChatOpenRouter(model="qwen/qwen-2.5-7b-instruct", max_tokens=5)
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

Response: The capital of France is Paris.
Usage Metadata: {'input_tokens': 36, 'output_tokens': 8, 'total_tokens': 44, 'input_token_details': {'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'reasoning': 0}}

--- Output Token Limit Example ---
Short Response: The history of the universe...
Finish Reason: length


"""