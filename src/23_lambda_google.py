"""
Title: RunnableLambda for Custom Processing (Google Generative AI)

RunnableLambda turns an ordinary Python function into a step of the chain,
so your own code can run at any point: before the model, between steps or
after the model.

uv run python src/23_lambda_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Custom preprocessing function
    def preprocess(text: str) -> dict:
        """Hide mobile numbers and emails before the text reaches the model."""
        # Example:
        #   input:  "Call me on 9876543210 or mail abhishek@example.com"
        #   prints: [before] Hidden 2 personal details. Text now: Call me on [MOBILE] or mail [EMAIL]
        #   output: {"text": "Call me on [MOBILE] or mail [EMAIL]", "masked_count": 2}
        words = text.strip().split()  # split the message into words
        safe_words = []
        hidden_count = 0

        for word in words:
            if "@" in word:  # an email address has an @ in it
                safe_words.append("[EMAIL]")
                hidden_count += 1
            elif word.isdigit() and len(word) == 10:  # a mobile number is 10 digits
                safe_words.append("[MOBILE]")
                hidden_count += 1
            else:
                safe_words.append(word)

        safe_text = " ".join(safe_words)
        print(f"[before] Hidden {hidden_count} personal details. Text now: {safe_text}")
        return {"text": safe_text, "masked_count": hidden_count}

    # Custom middle function (between the model and the parser)
    def show_token_usage(message):
        """Print the tokens used by the call and pass the reply on."""
        # The model's reply is an object, not plain text. It holds the answer
        # (message.content) and the token counts (message.usage_metadata).
        # Example:
        #   input:  the reply object
        #   prints: [middle] Tokens used: 52 in, 18 out
        #   output: the same reply object
        usage = message.usage_metadata or {}
        print(f"[middle] Tokens used: {usage.get('input_tokens')} in, {usage.get('output_tokens')} out")
        return message

    # Custom postprocessing function
    def format_output(result: str) -> str:
        """Add a heading above the answer."""
        # Example:
        #   input:  "The customer reports a failed UPI payment."
        #   prints: [after] Adding a heading to an answer of 7 words
        #   output: Support Message Summary (personal data hidden):
        #           ----------------------------------------
        #           The customer reports a failed UPI payment.
        print(f"[after] Adding a heading to an answer of {len(result.split())} words")
        line = "-" * 40  # a line made of 40 dashes
        return f"Support Message Summary (personal data hidden):\n{line}\n{result}"

    # Build chain with custom functions
    # masked_count tells the model that the placeholders hide real details,
    # so it should not guess them.
    prompt = ChatPromptTemplate.from_template(
        "Summarize this support message in one sentence.\n"
        "{masked_count} personal details were replaced with placeholders "
        "like [MOBILE] or [EMAIL]. Do not guess the hidden details.\n\n"
        "Message: {text}"
    )

    # The chain runs in this order (* marks our own functions):
    #   1. preprocess*       - hide mobile numbers and emails
    #   2. prompt            - build the prompt
    #   3. llm               - the model replies
    #   4. show_token_usage* - print the tokens used
    #   5. parser            - reply object to plain text
    #   6. format_output*    - add a heading
    chain = (
        RunnableLambda(preprocess)
        | prompt
        | llm
        | RunnableLambda(show_token_usage)
        | parser
        | RunnableLambda(format_output)
    )

    print("=" * 60)
    print("CUSTOM PROCESSING WITH LAMBDAS")
    print("=" * 60)

    result = chain.invoke(
        "  Hi, I am Abhishek. Please call me on 9876543210 or mail abhishek@example.com "
        "about my failed UPI payment of ₹4,999.  "
    )
    print(result)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
CUSTOM PROCESSING WITH LAMBDAS
============================================================
[before] Hidden 2 personal details. Text now: Hi, I am Abhishek. Please call me on [MOBILE] or mail [EMAIL] about my failed UPI payment of ₹4,999.
[middle] Tokens used: 67 in, 30 out
[after] Adding a heading to an answer of 19 words
Support Message Summary (personal data hidden):
----------------------------------------
Abhishek is requesting a callback at [MOBILE] or an email at [EMAIL] regarding a failed UPI payment of ₹4,999.


"""