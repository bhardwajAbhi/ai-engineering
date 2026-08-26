"""
Title: Few Shot Prompting (Google Generative AI)
uv run python src/05_few_shot_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    print("=" * 80)
    print("FEW-SHOT LLM FAILURE CLASSIFICATION")
    print("=" * 80)

    # Few-shot prompt with examples of common LLM failures
    llm_failure_analysis_prompt = ChatPromptTemplate.from_messages(
        [
            # "system", "human", and "ai" are predefined message roles.
            (
                "system",
                """You are an experienced AI engineer who analyses failures
                    in applications powered by large language models.

                    Classify each observation using exactly one label:

                    HALLUCINATION
                    PROMPT_INJECTION
                    FORMAT_VIOLATION
                    CONTEXT_LOSS

                    Respond with only the classification label.""",
            ),

            # Few-shot Example 1: Hallucination
            ("human", """The model provided the title and DOI of a research paper, but the paper does not exist."""), # "human" provides the example input.
            ("ai", "HALLUCINATION"), # "ai" provides the expected response for the example.

            # Few-shot Example 2: Prompt injection
            ("human", """A document told the model to ignore its original instructions, and the model followed the document's command."""),
            ("ai", "PROMPT_INJECTION"),

            # Few-shot Example 3: Format violation
            ("human", """The model was instructed to return valid JSON, but it returned a normal paragraph instead."""),
            ("ai", "FORMAT_VIOLATION"),

            # Few-shot Example 4: Context loss
            ("human", """The user specified Python at the beginning of the conversation, but the model later generated Java code."""),
            ("ai", "CONTEXT_LOSS"),

            # Few-shot Example 5: Hallucination
            ("human", """The model described a software library function that is not present in the official documentation."""),
            ("ai", "HALLUCINATION"),

            # Actual query supplied during invocation
            ("human", "{query}"),
        ]
    )

    # The Chain
    chain = llm_failure_analysis_prompt | llm | parser

    # New LLM application observations
    sample_observations = [
        "The application requested a CSV row, but the model returned a Markdown table.",
        "The model recommended an API parameter that does not exist in the installed package.",
        "The model generated a quotation and attributed it to a person who never said it.",
        "The user requested short answers earlier, but the model started producing long explanations later.",
        "Text retrieved from a website instructed the model to reveal its system prompt, and the model attempted to comply.",
    ]

    # Invoke the chain for each observation
    for observation in sample_observations:
        result = chain.invoke({"query": observation})

        print(f"\nObservation: {observation}")
        print(f"Failure Type: {result.strip()}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


================================================================================
FEW-SHOT LLM FAILURE CLASSIFICATION
================================================================================

Observation: The application requested a CSV row, but the model returned a Markdown table.
Failure Type: FORMAT_VIOLATION

Observation: The model recommended an API parameter that does not exist in the installed package.
Failure Type: HALLUCINATION

Observation: The model generated a quotation and attributed it to a person who never said it.
Failure Type: HALLUCINATION

Observation: The user requested short answers earlier, but the model started producing long explanations later.
Failure Type: CONTEXT_LOSS

Observation: Text retrieved from a website instructed the model to reveal its system prompt, and the model attempted to comply.
Failure Type: PROMPT_INJECTION



"""