"""
Title: RunnablePassthrough for Data Enrichment (Google Generative AI)

RunnablePassthrough.assign keeps the original input and adds a new field to
it, so a later step can use both the original values and the new result,
and we can print every step at the end.

uv run python src/21_passthrough_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
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

    # Compose: preserve input, add draft, then add critique
    # assign() keeps what the chain already has and adds one new field.
    # What the chain ends with decides what you get back, so choose by what you need:
    #   - ends with another chain: you get only that chain's output (a string)
    #   - ends with another assign(): you get everything so far (a dictionary)
    # This file ends with assign(), so the topic, draft and critique can all be printed.
    full_chain = (
        RunnablePassthrough.assign(draft=draft_chain)
        | RunnablePassthrough.assign(critique=critique_chain)
    )

    print("=" * 60)
    print("DRAFT THEN CRITIQUE PIPELINE")
    print("=" * 60)

    result = full_chain.invoke({"topic": "How an Android app should record and honor user consent under India's DPDP Act, 2023"})

    print("\nTOPIC:")
    print(result["topic"])

    print("\nDRAFT:")
    print(result["draft"])

    print("\nCRITIQUE:")
    print(result["critique"])


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

============================================================
DRAFT THEN CRITIQUE PIPELINE
============================================================

TOPIC:
How an Android app should record and honor user consent under India's DPDP Act, 2023

DRAFT:
To comply with India’s Digital Personal Data Protection (DPDP) Act, 2023, an Android app must implement a granular, clear, and affirmative consent mechanism that is presented in plain language, ideally in both English and the user’s preferred regional language. The app should provide a dedicated "Consent Manager" interface, allowing users to easily review, manage, or withdraw their consent at any time, with the withdrawal process being as simple as the initial granting process. Developers must ensure that consent is specific to a defined purpose, meaning data collection must be limited to what is strictly necessary for the stated service, and the app must maintain a robust, auditable record of when and how consent was obtained. Furthermore, the app should provide a clear "Notice" at the point of collection, detailing the types of data being processed and the user’s rights, ensuring that the entire lifecycle of data handling—from collection to potential deletion—remains transparent and fully under the user's control.

CRITIQUE:
Your draft provides a solid high-level overview of the requirements under the DPDP Act. However, to make it actionable for an Android development context, it needs to bridge the gap between legal theory and technical implementation.

Here are three specific improvements to enhance your draft:

### 1. Explicitly Address the "Consent Manager" Technical Requirement
The DPDP Act (Section 8(8)) introduces the concept of a **Consent Manager** as a specific entity registered with the Data Protection Board. Your draft currently uses the term to describe an in-app settings menu.
*   **Improvement:** Clarify that while an in-app "Privacy Dashboard" is a best practice for user experience, the app must also be architected to interface with the legal framework of a "Consent Manager" (the third-party entity). You should specify that the app must provide a mechanism for users to manage their consent either directly through the app or via a registered Consent Manager, ensuring the app can receive and act upon "withdrawals" sent by these entities.

### 2. Incorporate "Data Fiduciary" Obligations and Proof of Consent
The draft mentions maintaining an "auditable record," but it lacks the technical specificity required for compliance. Under the DPDP Act, the burden of proof lies with the Data Fiduciary (the app owner).
*   **Improvement:** Specify that the app must implement a **"Consent Log" architecture**. This should include capturing metadata such as the version of the Privacy Notice accepted, the timestamp, the specific device ID, and the exact scope of the consent granted. This log must be immutable and stored securely to serve as legal evidence in the event of an audit or a dispute, moving beyond a simple "true/false" flag in a database.

### 3. Define "Notice" Requirements (The "Itemized" Approach)
The draft mentions a "clear Notice," but the DPDP Act is very specific about what that notice must contain (e.g., the right to grievance redressal, the right to nominate, and the contact details of the Data Protection Officer).
*   **Improvement:** Add a requirement for **"Layered Notices."** Because mobile screens are small, you should suggest a "Just-in-Time" notice approach. The app should present a concise summary of the data being collected at the point of interaction (e.g., a pop-up before accessing the camera), with a link to the full, comprehensive Privacy Notice that includes the mandatory legal disclosures (DPO contact, grievance officer details, and the right to lodge a complaint with the Board).

***

### Summary of suggested revision for the "Notice" section:
> *"To comply with the DPDP Act, the app must implement a **layered notice strategy**: providing a 'Just-in-Time' summary at the point of collection, linked to a comprehensive Privacy Notice that explicitly lists the Data Protection Officer’s contact details, the user’s right to grievance redressal, and the right to nominate a representative. Furthermore, the app must maintain an immutable **Consent Log**—capturing the timestamp, notice version, and scope—to serve as verifiable proof of compliance for the Data Fiduciary."*

"""