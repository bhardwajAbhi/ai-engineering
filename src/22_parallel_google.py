"""
Title: RunnableParallel for Concurrent Analysis (Google Generative AI)

RunnableParallel runs several independent chains on the same input at the
same time and returns all their results together in one dictionary.

uv run python src/22_parallel_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Three independent analysis chains
    sentiment_chain = (
        ChatPromptTemplate.from_template("Classify sentiment as POSITIVE/NEGATIVE/NEUTRAL: {text}")
        | llm
        | parser
    )

    summary_chain = (
        ChatPromptTemplate.from_template("Summarize in one sentence: {text}") | llm | parser
    )

    language_chain = (
        ChatPromptTemplate.from_template(
            "What language is this text written in? Answer with just the language name: {text}"
        )
        | llm
        | parser
    )

    # Run all three in parallel
    parallel_chain = RunnableParallel(
        {
            "sentiment": sentiment_chain,
            "summary": summary_chain,
            "language": language_chain,
        }
    )

    text = "यह ऐप बार-बार मेरा SMS पढ़ने की अनुमति मांगता है। मुझे अपनी निजी जानकारी की चिंता है, इसलिए मैं इसे अनइंस्टॉल कर रहा हूँ।"

    print("=" * 60)
    print("PARALLEL EXECUTION")
    print("=" * 60)
    print(f"\nText: {text}\n")

    result = parallel_chain.invoke({"text": text})

    print(f"Sentiment: {result['sentiment']}")
    print(f"Summary: {result['summary']}")
    print(f"Language: {result['language']}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

============================================================
PARALLEL EXECUTION
============================================================

Text: यह ऐप बार-बार मेरा SMS पढ़ने की अनुमति मांगता है। मुझे अपनी निजी जानकारी की चिंता है, इसलिए मैं इसे अनइंस्टॉल कर रहा हूँ।

Sentiment: इसका सही वर्गीकरण **NEGATIVE** (नकारात्मक) है।
Summary: यह ऐप बार-बार SMS पढ़ने की अनुमति मांग रहा है, जिससे अपनी निजता (privacy) को सुरक्षित रखने के लिए मैं इसे अनइंस्टॉल कर रहा हूँ।
Language: Hindi

"""