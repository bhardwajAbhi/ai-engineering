"""
Title: Basic Chain Composition (OpenRouter)

A chain joins separate components (prompt, model and parser) with the pipe
operator, so the output of each one becomes the input of the next.

uv run python src/20_chain_composition_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Three components
    prompt = ChatPromptTemplate.from_template("Translate the following to Hinglish: {text}")
    model = ChatOpenRouter(model="meta-llama/llama-3.1-8b-instruct", temperature=0)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Use it
    result = chain.invoke(
        {"text": "This app needs permission to read your SMS only to fill in the OTP for you. It does not store your messages."}
    )
    print(f"Translation: {result}")

    # See intermediate types
    print(f"\nPrompt output type: {type(prompt.invoke({'text': 'test'}))}")
    print(f"Model output type: {type(model.invoke('test'))}")
    print(f"Chain output type: {type(result)}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

Translation: Is app ko permission chahiye sirf aapki SMS padhne ke liye, taaki OTP khud bhar jaaye. Ye aapki messages store nahi karta.

Prompt output type: <class 'langchain_core.prompt_values.ChatPromptValue'>
Model output type: <class 'langchain_core.messages.ai.AIMessage'>
Chain output type: <class 'langchain_core.messages.base.TextAccessor'>

"""