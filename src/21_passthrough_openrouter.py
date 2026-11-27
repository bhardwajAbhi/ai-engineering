"""
Title: RunnablePassthrough for Data Enrichment (OpenRouter)

RunnablePassthrough.assign keeps the original input and adds a new field to
it, so a later step can use both the original values and the new result.

uv run python src/21_passthrough_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Draft chain
    draft_prompt = ChatPromptTemplate.from_template("Write a one-paragraph summary of: {topic}")
    draft_chain = draft_prompt | llm | parser

    # Critique chain (needs both topic AND draft)
    critique_prompt = ChatPromptTemplate.from_template("""
        Topic: {topic}
        Draft: {draft}

        Critique this draft. List 3 specific improvements.
    """)
    critique_chain = critique_prompt | llm | parser

    # Compose: preserve input, add draft, then critique
    # assign() keeps what the chain already has and adds one new field.
    # What the chain ends with decides what you get back, so choose by what you need:
    #   - ends with another chain: you get only that chain's output (a string)
    #   - ends with another assign(): you get everything so far (a dictionary)
    # This file ends with a chain, so only the critique is returned.
    full_chain = RunnablePassthrough.assign(draft=draft_chain) | critique_chain

    print("=" * 60)
    print("DRAFT THEN CRITIQUE PIPELINE")
    print("=" * 60)

    result = full_chain.invoke({"topic": "SIM swap fraud and why SMS OTP login is at risk"})
    print(result)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

============================================================
DRAFT THEN CRITIQUE PIPELINE
============================================================
Here are 3 specific improvements for the draft:

1. **Expand the social engineering attack surface beyond carrier employees.** The draft only mentions persuading or bribing carrier staff, but SIM swap fraud frequently begins with attackers harvesting the victim's personal information (e.g., via phishing, data breaches, or social media) to answer security questions and initiate the swap themselves. Including this step would give readers a more complete picture of how attackers actually execute these schemes and what personal data they target first.

2. **Add concrete real-world impact or statistics.** The draft remains entirely abstract and theoretical. Including a specific example—such as the estimated billions of dollars lost to SIM swap fraud annually, or a notable case like the 2019 Twitter CEO account takeover—would ground the explanation in tangible stakes and make the threat feel more urgent and credible to the reader.

3. **Include actionable mitigations and alternatives to SMS OTP.** The draft thoroughly explains the vulnerability but stops there, leaving the reader without guidance on how to protect themselves. Adding a brief section on stronger alternatives—such as authenticator apps (TOTP), hardware security keys (FIDO2/WebAuthn), or carrier PINs/port-out locks—would transform the piece from a purely descriptive explanation into a more useful and balanced resource that empowers readers to take action.

"""