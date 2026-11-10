"""
Title: Self-Reflection Prompting (Google Generative AI)

Self-reflection asks the model to write a draft answer, then read that
draft as a reviewer and list specific improvements, and finally rewrite
the answer using that critique. Naming what is wrong first gives the
rewrite something concrete to fix.

uv run python src/15_self_reflection_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    # Temperature 0 keeps the draft, critique and revision steady.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    print("=" * 60)
    print("SELF-REFLECTION: DRAFT -> CRITIQUE -> REVISE")
    print("=" * 60)

    # Step 1: Draft
    draft_prompt = ChatPromptTemplate.from_template(
        "Provide a concise answer to the following question:\n\n{question}"
    )

    # Step 2: Critique
    critique_prompt = ChatPromptTemplate.from_template("""
        Critique the following answer based on these criteria:
        1. Is it factually correct?
        2. Is it clear and concise?
        3. Is it complete (covers all important aspects)?
        4. Is it well-structured?

        Provide specific, actionable improvements.

        QUESTION: {question}

        ANSWER TO CRITIQUE:
        {draft}

        CRITIQUE:
    """)

    # Step 3: Revise
    revise_prompt = ChatPromptTemplate.from_template("""
        Revise the original answer based on the provided critique.
        Make it better while keeping it concise.

        ORIGINAL QUESTION: {question}
        ORIGINAL ANSWER: {draft}
        CRITIQUE: {critique}

        REVISED ANSWER:
    """)

    # Create the three chains
    draft_chain = draft_prompt | llm | parser
    critique_chain = critique_prompt | llm | parser
    revise_chain = revise_prompt | llm | parser

    # Sample questions on DPDP and privacy, each with a length limit
    questions = [
        "Explain what a data fiduciary is under India's DPDP Act, 2023, in 3 sentences.",
        "Explain verifiable parental consent under India's DPDP Act, 2023, in 3 sentences.",
        "Explain purpose limitation for an app that collects personal data, in 3 sentences.",
    ]

    # Run the three steps for each question, one at a time
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"QUESTION {i}: {question}")
        print("=" * 60)

        # Step 1: write the draft and show it
        draft = draft_chain.invoke({"question": question})
        print(f"\n--- DRAFT ---\n{draft}")

        # Step 2: critique the draft and show it
        critique = critique_chain.invoke({"question": question, "draft": draft})
        print(f"\n--- CRITIQUE ---\n{critique}")

        # Step 3: revise the draft using the critique
        revised = revise_chain.invoke(
            {"question": question, "draft": draft, "critique": critique}
        )
        print(f"\n--- REVISED ANSWER ---\n{revised}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
SELF-REFLECTION: DRAFT -> CRITIQUE -> REVISE
============================================================

============================================================
QUESTION 1: Explain what a data fiduciary is under India's DPDP Act, 2023, in 3 sentences.
============================================================

--- DRAFT ---
Under India’s DPDP Act, 2023, a Data Fiduciary is any person or entity that determines the purpose and means of processing an individual's personal data. They are legally responsible for ensuring compliance with the Act, including obtaining valid consent and implementing robust security safeguards. Additionally, they must manage the data processing activities of any third-party "Data Processors" they engage to handle information on their behalf.

--- CRITIQUE ---
### Critique

**1. Is it factually correct?**
**Yes.** The definition accurately reflects the role of a Data Fiduciary as defined in Section 2(i) of the Digital Personal Data Protection (DPDP) Act, 2023.

**2. Is it clear and concise?**
**Yes.** The language is professional, accessible, and avoids unnecessary jargon. It adheres well to the three-sentence constraint.

**3. Is it complete?**
**Mostly.** It covers the core definition, the primary obligations (compliance/security), and the relationship with Data Processors. However, it misses a critical nuance: the distinction between a "Data Fiduciary" and a "Significant Data Fiduciary" (SDF). Given the importance of SDFs under the Act, mentioning that certain entities may face stricter obligations would have made the answer more comprehensive.

**4. Is it well-structured?**
**Yes.** The flow is logical: Definition $\rightarrow$ Primary Responsibility $\rightarrow$ Relationship with third parties.

---

### Actionable Improvements

*   **Incorporate the "Significant" distinction:** To make the answer more complete, briefly acknowledge that the Act creates a sub-category for entities with higher risk or volume.
*   **Refine the third sentence:** While the mention of "Data Processors" is accurate, the Act emphasizes that the Data Fiduciary remains *accountable* for the actions of the processor. Using the word "accountable" aligns better with the legal language of the Act.

**Suggested Revised Version:**
> Under India’s DPDP Act, 2023, a Data Fiduciary is any person or entity that determines the purpose and means of processing an individual's personal data. They are legally responsible for ensuring compliance with the Act, including obtaining valid consent and implementing robust security safeguards. Furthermore, they remain accountable for the actions of any third-party "Data Processors" they engage, and may be classified as a "Significant Data Fiduciary" if they handle high volumes of sensitive data.

--- REVISED ANSWER ---
Under India’s DPDP Act, 2023, a Data Fiduciary is any person or entity that determines the purpose and means of processing an individual's personal data. They are legally responsible for ensuring compliance, including obtaining valid consent and implementing robust security safeguards. Furthermore, they remain accountable for the actions of any third-party "Data Processors" they engage and may be classified as a "Significant Data Fiduciary" if they handle high volumes of data or pose specific risks.

============================================================
QUESTION 2: Explain verifiable parental consent under India's DPDP Act, 2023, in 3 sentences.
============================================================

--- DRAFT ---
Under the DPDP Act, 2023, data fiduciaries must obtain verifiable parental consent before processing the personal data of children (individuals under 18). This consent must be free, specific, informed, unconditional, and unambiguous, provided by a parent or lawful guardian. Additionally, the Act prohibits fiduciaries from undertaking tracking, behavioral monitoring, or targeted advertising directed at children.

--- CRITIQUE ---
### Critique

**1. Is it factually correct?**
**Yes.** The answer accurately identifies the requirement for verifiable parental consent, the age threshold (under 18), the nature of the consent, and the specific prohibitions regarding tracking and advertising.

**2. Is it clear and concise?**
**Yes.** The language is professional, and the information is presented without unnecessary jargon. It adheres well to the three-sentence constraint.

**3. Is it complete?**
**Mostly.** While it covers the core requirements, it misses a critical nuance: the Act specifies that the government may exempt certain classes of data fiduciaries from these requirements or set a lower age of consent for specific services. Additionally, it does not explicitly mention that the "verifiable" aspect implies the fiduciary must take reasonable steps to confirm the identity of the parent/guardian.

**4. Is it well-structured?**
**Yes.** The flow is logical: it defines the obligation, describes the quality of consent, and lists the associated prohibitions.

---

### Specific, Actionable Improvements

*   **Clarify the "Verifiable" aspect:** The current answer mentions "verifiable" but doesn't explain that the fiduciary is responsible for the *process* of verification. Adding a brief mention of "reasonable efforts" would strengthen the legal accuracy.
*   **Address the "Exemption" nuance:** To be truly complete, acknowledge that the Act allows for potential government-notified exemptions or age-limit variations, as this is a significant feature of the DPDP Act.
*   **Refine for impact:** You can combine the "nature of consent" and "prohibitions" more tightly to make room for the nuance regarding the government's power to regulate these rules.

**Suggested Revised Version:**
> Under the DPDP Act, 2023, data fiduciaries must obtain verifiable consent from a parent or lawful guardian before processing the personal data of any child under 18. This consent must be free, specific, informed, unconditional, and unambiguous, with fiduciaries required to take reasonable steps to verify the identity of the consenting adult. Furthermore, the Act strictly prohibits fiduciaries from tracking, behavioral monitoring, or targeted advertising directed at children, subject to any government-notified exemptions or age-limit variations.

--- REVISED ANSWER ---
Under the DPDP Act, 2023, data fiduciaries must obtain verifiable consent from a parent or lawful guardian before processing the personal data of any child under 18. Fiduciaries are required to take reasonable steps to verify the identity of the consenting adult, ensuring the consent is free, specific, informed, unconditional, and unambiguous. Furthermore, the Act strictly prohibits tracking, behavioral monitoring, or targeted advertising directed at children, subject to any government-notified exemptions or age-limit variations.

============================================================
QUESTION 3: Explain purpose limitation for an app that collects personal data, in 3 sentences.
============================================================

--- DRAFT ---
Purpose limitation requires that an app only collects personal data for specified, explicit, and legitimate purposes defined at the time of collection. Once collected, the data must not be processed or used in any way that is incompatible with those original stated intentions. This principle ensures transparency and protects users by preventing apps from repurposing sensitive information without further consent.

--- CRITIQUE ---
### Critique

**1. Is it factually correct?**
Yes. The answer accurately reflects the core tenets of the "purpose limitation" principle as defined in major data protection frameworks like the GDPR.

**2. Is it clear and concise?**
Yes. The language is professional, accessible, and avoids unnecessary jargon. It adheres strictly to the three-sentence constraint.

**3. Is it complete?**
It is mostly complete, but it misses a subtle but important nuance: the concept of "data minimization" is often linked to purpose limitation (i.e., you shouldn't collect *more* data than is necessary for those specific purposes). However, for a three-sentence summary, it covers the essential legal and ethical requirements well.

**4. Is it well-structured?**
Yes. The flow is logical: it defines the requirement, explains the restriction on future use, and concludes with the underlying benefit (transparency/protection).

---

### Actionable Improvements

The answer is already very strong. To make it even more precise, consider these minor adjustments:

*   **Refine the second sentence for legal precision:** While "incompatible" is the correct term, you could explicitly mention that further processing is only allowed if it is *compatible* with the original purpose.
*   **Strengthen the "Why":** The current third sentence is good, but you could emphasize that this principle prevents "function creep," which is a common term used in privacy compliance.

**Suggested Revised Version:**
> Purpose limitation requires that an app collects personal data only for specified, explicit, and legitimate purposes defined at the time of collection. Once gathered, the data must not be processed in any manner incompatible with those original intentions. This principle protects users by preventing "function creep," ensuring that apps cannot repurpose sensitive information without obtaining further consent.

--- REVISED ANSWER ---
Purpose limitation requires that an app collects personal data only for specified, explicit, and legitimate purposes defined at the time of collection. Once gathered, the data must not be processed in any manner incompatible with those original intentions. This principle protects users by preventing "function creep," ensuring that apps cannot repurpose sensitive information without obtaining further consent.



"""