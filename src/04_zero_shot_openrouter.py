"""
Title: Zero Shot Prompting (OpenRouter)
uv run python src/04_zero_shot_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )
    
    # Parser to parse the LLM output
    parser = StrOutputParser()

    print("=" * 80)
    print("ZERO-SHOT CYBERSECURITY CLASSIFICATION")
    print("=" * 80)

    # Zero-shot security event classification prompt
    security_prompt = ChatPromptTemplate.from_template("""
        Classify the following security event as PHISHING, MALWARE, or BENIGN.
        Respond with only the classification label.

        Security event: {event}

        Classification:""")

    # The Chain
    chain = security_prompt | llm | parser

    # Sample cybersecurity events
    sample_events = [
        "An email asks the person to click a link to unlock his bank account.",
        "An unknown program is encrypting files on a computer.",
        "An antivirus scan completed and found no threats.",
        "An unknown sender sent an invoice with a suspicious attachment.",
        "A user logged in from their usual device.",
    ]
    
    # Invoking the Chain for each event and classifying
    for event in sample_events:
        result = chain.invoke({"event": event})
        print(f"\nEvent: {event}")
        print(f"Classification: {result.strip()}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


================================================================================
ZERO-SHOT CYBERSECURITY CLASSIFICATION
================================================================================

Event: An email asks the person to click a link to unlock his bank account.
Classification: PHISHING

Event: An unknown program is encrypting files on a computer.
Classification: MALWARE

Event: An antivirus scan completed and found no threats.
Classification: BENIGN

Event: An unknown sender sent an invoice with a suspicious attachment.
Classification: PHISHING

Event: A user logged in from their usual device.
Classification: BENIGN



"""