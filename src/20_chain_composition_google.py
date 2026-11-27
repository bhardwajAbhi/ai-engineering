"""
Title: Basic Chain Composition (Google Generative AI)

A chain joins separate components (prompt, model and parser) with the pipe
operator, so the output of each one becomes the input of the next.

uv run python src/20_chain_composition_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Three components
    prompt = ChatPromptTemplate.from_template("Translate the following to Hindi: {text}")
    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)
    parser = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parser

    # Use it
    result = chain.invoke(
        {"text": "We collect your mobile number only to send your login OTP. You can withdraw your consent at any time."}
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

Translation: यहाँ आपके वाक्य का हिंदी अनुवाद है:

**"हम आपका मोबाइल नंबर केवल लॉगिन OTP भेजने के लिए एकत्र करते हैं। आप किसी भी समय अपनी सहमति वापस ले सकते हैं।"**

Prompt output type: <class 'langchain_core.prompt_values.ChatPromptValue'>
Model output type: <class 'langchain_core.messages.ai.AIMessage'>
Chain output type: <class 'langchain_core.messages.base.TextAccessor'>

"""