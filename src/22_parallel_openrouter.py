"""
Title: RunnableParallel for Concurrent Analysis (OpenRouter)

RunnableParallel runs several independent chains on the same input at the
same time and returns all their results together in one dictionary.

uv run python src/22_parallel_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Three independent analysis chains
    verdict_chain = (
        ChatPromptTemplate.from_template("Classify this message as PHISHING, SUSPICIOUS or LEGITIMATE. Answer with just the label: {text}")
        | llm
        | parser
    )

    summary_chain = (
        ChatPromptTemplate.from_template("Summarize in one sentence: {text}") | llm | parser
    )

    trick_chain = (
        ChatPromptTemplate.from_template(
            "What trick does this message use to make the reader act fast? Answer in a few words: {text}"
        )
        | llm
        | parser
    )

    # Run all three in parallel
    parallel_chain = RunnableParallel(
        {
            "verdict": verdict_chain,
            "summary": summary_chain,
            "trick": trick_chain,
        }
    )

    text = "Dear customer, your bank account will be blocked today. Update your KYC now at bit.ly/kyc-upd8 or lose access to your money."

    print("=" * 60)
    print("PARALLEL EXECUTION")
    print("=" * 60)
    print(f"\nText: {text}\n")

    result = parallel_chain.invoke({"text": text})

    print(f"Verdict: {result['verdict']}")
    print(f"Summary: {result['summary']}")
    print(f"Trick: {result['trick']}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

============================================================
PARALLEL EXECUTION
============================================================

Text: Dear customer, your bank account will be blocked today. Update your KYC now at bit.ly/kyc-upd8 or lose access to your money.

Verdict: PHISHING
Summary: This message is a phishing scam that uses false urgency and a suspicious shortened link to trick recipients into revealing their banking credentials under the pretense of a KYC update.
Trick: Urgency and fear of loss.

"""