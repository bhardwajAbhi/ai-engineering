"""
Title: Chain of Verification Prompting (Google Generative AI)
Chain of Verification asks the model to write an answer, pull out the
facts it stated, check each fact on its own and then fix any fact found
to be wrong. Checking small facts one at a time is easier than judging
a whole answer at once.
uv run python src/14_chain_of_verification_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def chain_of_verification(
    question: str,
    generate_chain,
    extract_chain,
    verify_chain,
    revise_chain,
) -> dict:
    """Run the full Chain of Verification pipeline."""

    # Step 1: Generate the initial response
    initial_response = generate_chain.invoke({"question": question})
    print("INITIAL RESPONSE:")
    print(initial_response)
    print("\n" + "-" * 40)

    # Step 2: Extract the claims, one per line
    claims_text = extract_chain.invoke({"response": initial_response})
    claims = [c.strip() for c in claims_text.strip().split("\n") if c.strip()]
    print("EXTRACTED CLAIMS:")
    for claim in claims:
        print(f"  {claim}")
    print("\n" + "-" * 40)

    # Step 3: Verify each claim on its own
    verification_results = []
    print("VERIFICATION:")
    for claim in claims:
        result = verify_chain.invoke({"claim": claim})
        verification_results.append(f"{claim}\n  -> {result.strip()}")
        print(f"  {claim}")
        print(f"    -> {result.strip()}")
    print("\n" + "-" * 40)

    # Step 4: Decide what to do, based on the verification results
    verification_text = "\n".join(verification_results)

    if "INCORRECT" in verification_text:
        # At least one claim is wrong, so rewrite the response
        final_response = revise_chain.invoke(
            {"response": initial_response, "verification_results": verification_text}
        )
        print("REVISED RESPONSE:")
        print(final_response)
    elif "UNCERTAIN" in verification_text:
        # Nothing is known to be wrong, but some claims could not be checked
        final_response = initial_response
        print("NO REVISION NEEDED - but some claims are UNCERTAIN")
        print("The initial response is kept, with unchecked claims in it.")
    else:
        # Every claim was verified
        final_response = initial_response
        print("NO REVISION NEEDED - All claims verified")

    return {
        "initial": initial_response,
        "claims": claims,
        "verifications": verification_results,
        "final": final_response,
    }


def main():
    load_dotenv()

    # Initialize the model
    # Temperature 0 keeps the checks steady and repeatable.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Step 1 prompt: generate the initial response
    generate_prompt = ChatPromptTemplate.from_template("""
        Answer this question with specific facts and details:
        {question}
    """)

    # Step 2 prompt: extract the claims
    extract_claims_prompt = ChatPromptTemplate.from_template("""
        Extract all specific factual claims from this text.
        List each claim on a separate line, numbered.
        Focus on dates, numbers, names and verifiable facts.

        TEXT:
        {response}

        FACTUAL CLAIMS:
    """)

    # Step 3 prompt: verify one claim
    verify_prompt = ChatPromptTemplate.from_template("""
        Verify this specific claim. Is it accurate?

        CLAIM: {claim}

        Think carefully. If you are not certain, say "UNCERTAIN".
        Respond with exactly one of these:
        - VERIFIED: if you are confident the claim is correct
        - INCORRECT: <the correct information>, if you know the claim is wrong
        - UNCERTAIN: if you are not sure
    """)

    # Step 4 prompt: revise the response
    revise_prompt = ChatPromptTemplate.from_template("""
        Revise this response to fix any incorrect claims.

        ORIGINAL RESPONSE:
        {response}

        VERIFICATION RESULTS:
        {verification_results}

        REVISED RESPONSE (fix any incorrect claims, keep verified ones):
    """)

    # Create the four chains
    generate_chain = generate_prompt | llm | parser
    extract_chain = extract_claims_prompt | llm | parser
    verify_chain = verify_prompt | llm | parser
    revise_chain = revise_prompt | llm | parser

    # A question full of dates, numbers and names, so there is
    # more to check and more chance of a slip
    question = (
        "Tell me about India's Digital Personal Data Protection (DPDP) Act, "
        "2023: when it became law, who it applies to, the rights it gives "
        "people, and the penalties it sets."
    )

    print("=" * 60)
    print("QUESTION")
    print("=" * 60)
    print(f"\n{question}\n")
    print("=" * 60 + "\n")

    chain_of_verification(
        question, generate_chain, extract_chain, verify_chain, revise_chain
    )


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
QUESTION
============================================================

Tell me about India's Digital Personal Data Protection (DPDP) Act, 2023: when it became law, who it applies to, the rights it gives people, and the penalties it sets.

============================================================

INITIAL RESPONSE:
India’s **Digital Personal Data Protection (DPDP) Act, 2023** represents a significant overhaul of the country’s data privacy framework. It was enacted to provide a comprehensive legal structure for the processing of digital personal data.

Here are the specific facts and details regarding the Act:

### 1. When it became law
*   **Passage:** The Act was passed by both houses of the Indian Parliament in early August 2023.
*   **Presidential Assent:** It received the assent of the President of India on **August 11, 2023**, officially becoming law.
*   **Implementation Status:** While the Act is law, it is being implemented in a phased manner. As of mid-2024, the government is in the process of notifying the specific rules and establishing the Data Protection Board of India, which will oversee enforcement.

### 2. Who it applies to
The Act applies to the processing of **digital personal data** within India. Its scope includes:
*   **Personal data collected in digital form:** This includes data collected offline that is subsequently digitized.
*   **Extraterritorial reach:** It applies to the processing of personal data outside India if such processing is in connection with any activity related to offering goods or services to individuals (Data Principals) located within India.
*   **Data Fiduciaries:** The Act places obligations on "Data Fiduciaries"—entities (individuals, companies, or government bodies) that determine the purpose and means of processing personal data.

### 3. Rights of the Data Principal (Individuals)
The Act grants individuals (referred to as "Data Principals") several key rights:
*   **Right to Information:** The right to obtain a summary of the personal data being processed and the identities of all other Data Fiduciaries with whom the data has been shared.
*   **Right to Correction and Erasure:** The right to demand the correction of inaccurate data, completion of incomplete data, and the erasure of data that is no longer necessary for the purpose for which it was processed.
*   **Right to Grievance Redressal:** The right to have a readily available grievance redressal mechanism provided by the Data Fiduciary.
*   **Right to Nominate:** The right to nominate another individual to exercise their rights in the event of death or incapacity.

### 4. Penalties
The Act introduces a tiered penalty structure for non-compliance, which are among the highest in Indian regulatory history:
*   **Failure to prevent data breaches:** Up to **₹250 crore** (approx. $30 million USD).
*   **Failure to notify the Board or affected individuals of a data breach:** Up to **₹200 crore** (approx. $24 million USD).
*   **Failure to comply with additional obligations regarding children:** Up to **₹200 crore** (approx. $24 million USD).
*   **General breaches:** For other contraventions of the Act, penalties can reach up to **₹50 crore** (approx. $6 million USD).

**Key Enforcement Mechanism:**
The Act establishes the **Data Protection Board of India**. This independent body is tasked with investigating data breaches, hearing complaints from individuals, and imposing the financial penalties mentioned above. Unlike some international laws (like the GDPR), the DPDP Act does not specify a percentage of global turnover for penalties, opting instead for fixed maximum monetary caps.

----------------------------------------
EXTRACTED CLAIMS:
  1. The name of the legislation is the Digital Personal Data Protection (DPDP) Act, 2023.
  2. The Act was passed by both houses of the Indian Parliament in early August 2023.
  3. The Act received the assent of the President of India on August 11, 2023.
  4. As of mid-2024, the government is in the process of notifying specific rules and establishing the Data Protection Board of India.
  5. The Act applies to the processing of digital personal data within India.
  6. The Act applies to personal data collected offline that is subsequently digitized.
  7. The Act has extraterritorial reach regarding the processing of personal data outside India if it relates to offering goods or services to individuals located within India.
  8. The Act defines "Data Fiduciaries" as entities (individuals, companies, or government bodies) that determine the purpose and means of processing personal data.
  9. The Act grants individuals the right to obtain a summary of personal data being processed and the identities of other Data Fiduciaries with whom the data is shared.
  10. The Act grants individuals the right to demand the correction of inaccurate data, completion of incomplete data, and the erasure of data no longer necessary.
  11. The Act grants individuals the right to a grievance redressal mechanism provided by the Data Fiduciary.
  12. The Act grants individuals the right to nominate another person to exercise their rights in the event of death or incapacity.
  13. The maximum penalty for failure to prevent data breaches is ₹250 crore (approx. $30 million USD).
  14. The maximum penalty for failure to notify the Board or affected individuals of a data breach is ₹200 crore (approx. $24 million USD).
  15. The maximum penalty for failure to comply with obligations regarding children is ₹200 crore (approx. $24 million USD).
  16. The maximum penalty for general breaches of the Act is ₹50 crore (approx. $6 million USD).
  17. The Data Protection Board of India is the independent body tasked with investigating breaches, hearing complaints, and imposing financial penalties.
  18. The DPDP Act uses fixed maximum monetary caps for penalties rather than a percentage of global turnover.

----------------------------------------
VERIFICATION:
  1. The name of the legislation is the Digital Personal Data Protection (DPDP) Act, 2023.
    -> VERIFIED: The name of the legislation is the Digital Personal Data Protection (DPDP) Act, 2023.
  2. The Act was passed by both houses of the Indian Parliament in early August 2023.
    -> Please provide the name of the "Act" you are referring to. Without the specific title of the legislation, I cannot verify the claim.
  3. The Act received the assent of the President of India on August 11, 2023.
    -> Please provide the name of the Act you are referring to. Without the title of the legislation, it is impossible to verify the date of presidential assent.
  4. As of mid-2024, the government is in the process of notifying specific rules and establishing the Data Protection Board of India.
    -> VERIFIED
  5. The Act applies to the processing of digital personal data within India.
    -> INCORRECT: The Act applies to the processing of digital personal data within India, AND to the processing of digital personal data outside India if such processing is in connection with any activity related to offering goods or services to data principals within the territory of India.
  6. The Act applies to personal data collected offline that is subsequently digitized.
    -> Please provide the name of the specific "Act" you are referring to (e.g., GDPR, CCPA, VCDPA, etc.). Without the name of the legislation, it is impossible to verify the claim.

**UNCERTAIN**
  7. The Act has extraterritorial reach regarding the processing of personal data outside India if it relates to offering goods or services to individuals located within India.
    -> VERIFIED: The Digital Personal Data Protection Act, 2023 (DPDP Act) includes Section 3(a), which explicitly states that the Act applies to the processing of digital personal data outside the territory of India if such processing is in connection with any activity related to offering goods or services to Data Principals within the territory of India.
  8. The Act defines "Data Fiduciaries" as entities (individuals, companies, or government bodies) that determine the purpose and means of processing personal data.
    -> VERIFIED
  9. The Act grants individuals the right to obtain a summary of personal data being processed and the identities of other Data Fiduciaries with whom the data is shared.
    -> VERIFIED
  10. The Act grants individuals the right to demand the correction of inaccurate data, completion of incomplete data, and the erasure of data no longer necessary.
    -> UNCERTAIN
  11. The Act grants individuals the right to a grievance redressal mechanism provided by the Data Fiduciary.
    -> Please specify which "Act" you are referring to (e.g., the Digital Personal Data Protection Act, 2023 of India, or another specific legislation). Without the name of the Act, I cannot verify the claim.

**UNCERTAIN**
  12. The Act grants individuals the right to nominate another person to exercise their rights in the event of death or incapacity.
    -> UNCERTAIN
  13. The maximum penalty for failure to prevent data breaches is ₹250 crore (approx. $30 million USD).
    -> INCORRECT: Under the Digital Personal Data Protection Act (DPDP Act) 2023 of India, the maximum penalty for failure to prevent a data breach (specifically failure to take reasonable security safeguards) is ₹250 crore per instance. However, the claim is incomplete because the Act specifies that the penalty is "up to ₹250 crore" for each instance of non-compliance, and there is no upper cap on the total aggregate penalty if multiple breaches or violations occur.
  14. The maximum penalty for failure to notify the Board or affected individuals of a data breach is ₹200 crore (approx. $24 million USD).
    -> INCORRECT: The maximum penalty for failure to notify the Data Protection Board or affected individuals of a data breach under the Digital Personal Data Protection Act, 2023, is ₹250 crore (approx. $30 million USD).
  15. The maximum penalty for failure to comply with obligations regarding children is ₹200 crore (approx. $24 million USD).
    -> INCORRECT: The maximum penalty for failure to comply with obligations regarding children under the Digital Personal Data Protection Act (DPDP Act), 2023, is ₹200 crore. However, the claim is misleading because the ₹200 crore figure is the maximum penalty for *any* failure to comply with the Act's provisions (Section 8(9)), not specifically for obligations regarding children. The specific penalty for failing to comply with obligations regarding children (Section 9) is also capped at ₹200 crore, but this is part of the general penalty structure rather than a unique maximum for that specific section.
  16. The maximum penalty for general breaches of the Act is ₹50 crore (approx. $6 million USD).
    -> INCORRECT: The maximum penalty for general breaches of the Digital Personal Data Protection Act (DPDPA) 2023 is ₹250 crore (approx. $30 million USD).
  17. The Data Protection Board of India is the independent body tasked with investigating breaches, hearing complaints, and imposing financial penalties.
    -> VERIFIED
  18. The DPDP Act uses fixed maximum monetary caps for penalties rather than a percentage of global turnover.
    -> INCORRECT: The DPDP Act (Digital Personal Data Protection Act, 2023) specifies monetary caps for penalties (up to ₹250 crore per instance), but it does not use a percentage of global turnover. Therefore, the claim that it uses fixed caps *rather than* a percentage of global turnover is technically correct in its description of the Act's mechanism, but the phrasing implies a contrast with other laws (like the GDPR) that do use turnover. However, as the Act strictly relies on fixed monetary caps and does not include a turnover-based calculation, the claim is **VERIFIED**.

----------------------------------------
REVISED RESPONSE:
### Revised Response: India’s Digital Personal Data Protection (DPDP) Act, 2023

India’s **Digital Personal Data Protection (DPDP) Act, 2023** represents a significant overhaul of the country’s data privacy framework. It was enacted to provide a comprehensive legal structure for the processing of digital personal data.

Here are the specific facts and details regarding the Act:

### 1. When it became law
*   **Passage:** The Act was passed by both houses of the Indian Parliament in early August 2023.
*   **Presidential Assent:** It received the assent of the President of India on **August 11, 2023**, officially becoming law.
*   **Implementation Status:** While the Act is law, it is being implemented in a phased manner. As of mid-2024, the government is in the process of notifying the specific rules and establishing the Data Protection Board of India.

### 2. Who it applies to
The Act applies to the processing of **digital personal data** within India. Its scope includes:
*   **Personal data collected in digital form:** This includes data collected offline that is subsequently digitized.
*   **Extraterritorial reach:** The Act applies to the processing of digital personal data within India, and to the processing of digital personal data outside India if such processing is in connection with any activity related to offering goods or services to Data Principals within the territory of India.
*   **Data Fiduciaries:** The Act places obligations on "Data Fiduciaries"—entities (individuals, companies, or government bodies) that determine the purpose and means of processing personal data.

### 3. Rights of the Data Principal (Individuals)
The Act grants individuals (referred to as "Data Principals") several key rights:
*   **Right to Information:** The right to obtain a summary of the personal data being processed and the identities of all other Data Fiduciaries with whom the data has been shared.
*   **Right to Correction and Erasure:** The right to demand the correction of inaccurate data, completion of incomplete data, and the erasure of data that is no longer necessary for the purpose for which it was processed.
*   **Right to Grievance Redressal:** The right to have a readily available grievance redressal mechanism provided by the Data Fiduciary.
*   **Right to Nominate:** The right to nominate another individual to exercise their rights in the event of death or incapacity.

### 4. Penalties
The Act introduces a tiered penalty structure for non-compliance. Unlike the GDPR, the DPDP Act does not use a percentage of global turnover, opting instead for fixed maximum monetary caps:
*   **Failure to take reasonable security safeguards to prevent data breaches:** Up to **₹250 crore**.
*   **Failure to notify the Board or affected individuals of a data breach:** Up to **₹250 crore**.
*   **Failure to comply with obligations regarding children:** Up to **₹200 crore**.
*   **General breaches:** For other contraventions of the Act, penalties can reach up to **₹250 crore**.

*Note: These penalties are "up to" the specified amounts per instance of non-compliance, and there is no upper cap on the total aggregate penalty if multiple breaches or violations occur.*

### 5. Enforcement Mechanism
The Act establishes the **Data Protection Board of India**. This independent body is tasked with investigating data breaches, hearing complaints from individuals, and imposing the financial penalties mentioned above.

"""