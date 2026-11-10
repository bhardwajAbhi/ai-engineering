"""
Title: Self-Reflection Prompting (Ollama)

Self-reflection asks the model to write a draft answer, review that
draft and list improvements, and then rewrite the answer using the review.

uv run python src/15_self_reflection_ollama.py
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def main():
    # Initialize the model
    llm = ChatOllama(
        model="gemma:2b",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    print("=" * 60)
    print("SELF-REFLECTION PIPELINE: DRAFT -> CRITIQUE -> REVISE")
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

    # Join the three chains into one pipeline, so we call it only once.
    # Each assign() step runs a chain and saves its answer under a name
    # (draft, then critique, then revised) next to the earlier values.
    # This way each step can use the answers of the steps before it.
    pipeline = (
        RunnablePassthrough.assign(draft=draft_chain)
        | RunnablePassthrough.assign(critique=critique_chain)
        | RunnablePassthrough.assign(revised=revise_chain)
    )

    # Shorter option: end with "| revise_chain" instead of the last assign().
    # Then the pipeline gives back only the final revised answer as text.

    # Sample questions on DPDP, Android and mobile security
    questions = [
        "Explain what withdrawing consent means for an app under India's DPDP Act, 2023, in 3 sentences.",
        "Explain what an Android runtime permission is and why it exists, in 3 sentences.",
        "Explain why TLS certificate validation matters for a mobile app, in 3 sentences.",
    ]

    # Run the pipeline once for each question
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"QUESTION {i}: {question}")
        print("=" * 60)

        # One call runs all three steps and gives back every part
        result = pipeline.invoke({"question": question})

        print(f"\n--- DRAFT ---\n{result['draft']}")
        print(f"\n--- CRITIQUE ---\n{result['critique']}")
        print(f"\n--- REVISED ANSWER ---\n{result['revised']}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
SELF-REFLECTION PIPELINE: DRAFT -> CRITIQUE -> REVISE
============================================================

============================================================
QUESTION 1: Explain what withdrawing consent means for an app under India's DPDP Act, 2023, in 3 sentences.
============================================================

--- DRAFT ---
Sure, here's a concise answer to the question:

Withdrawing consent under the Indian Digital Personal Data Protection Act, 2023 allows an app to stop processing personal data and stop using the app.

--- CRITIQUE ---
**1. Is it factually correct?**
Yes, the answer is factually correct. The Indian Digital Personal Data Protection Act, 2023 does allow an app to withdraw consent and stop processing personal data.

**2. Is it clear and concise?**
Yes, the answer is clear and concise. It provides a concise definition of withdrawing consent in simple terms.

**3. Is it complete (covers all important aspects)?**
Yes, the answer covers all the important aspects of withdrawing consent, including the app's ability to stop processing personal data and the user's right to withdraw consent.

**4. Is it well-structured?**
Yes, the answer is well-structured and easy to understand. It uses simple language and provides a clear definition of withdrawing consent.

--- REVISED ANSWER ---
Sure, here's the revised answer:

Withdrawing consent under the Indian Digital Personal Data Protection Act, 2023 allows an app to stop processing personal data and stop using the app. This gives users greater control over their data and gives them the right to withdraw consent at any time.

============================================================
QUESTION 2: Explain what an Android runtime permission is and why it exists, in 3 sentences.
============================================================

--- DRAFT ---
Sure, here's a concise explanation of an Android runtime permission:

An Android runtime permission is a mechanism that allows an app to access specific resources or functionalities on the user's device. It is granted by the system based on the declared permissions in the app's manifest file.

--- CRITIQUE ---
**1. Is it factually correct?**
Yes, the answer is factually correct. It provides a clear definition of an Android runtime permission in 3 sentences.

**2. Is it clear and concise?**
Yes, the answer is clear and concise. It uses simple and straightforward language to explain the concept, making it easy for a broad audience to understand.

**3. Is it complete (covers all important aspects)?**
Yes, the answer covers all the important aspects of an Android runtime permission, including its purpose, how it is granted, and the resources it can access.

**4. Is it well-structured?**
Yes, the answer is well-structured and easy to follow. It uses a clear and logical format, with a concise explanation followed by a few examples.

**Recommendations for improvement:**

- Provide a few real-world examples of how runtime permissions are used.
- Explain the potential consequences of not requesting a specific permission.
- Provide a brief overview of the related concept of permissions and security.

--- REVISED ANSWER ---
Sure, here's the revised answer:

An Android runtime permission is a mechanism that allows an app to access specific resources or functionalities on the user's device with explicit user consent. It is granted by the system based on the declared permissions in the app's manifest file. Understanding and respecting runtime permissions is crucial for ensuring app security and user privacy.

============================================================
QUESTION 3: Explain why TLS certificate validation matters for a mobile app, in 3 sentences.
============================================================

--- DRAFT ---
TLS certificate validation ensures the authenticity and trustworthiness of the app's communication channels, protecting sensitive user data and ensuring a secure connection.

--- CRITIQUE ---
**1. Is it factually correct?**
Yes, the answer is factually correct. TLS certificate validation is a crucial security measure for mobile apps to ensure the authenticity and trustworthiness of the communication channels they use.

**2. Is it clear and concise?**
Yes, the answer is clear and concise. It provides a brief yet comprehensive explanation of the importance of TLS certificate validation for mobile app security.

**3. Is it complete (covers all important aspects)?**
Yes, the answer covers all the important aspects of TLS certificate validation, including the role of certificates in authenticating the app, the protection of sensitive user data, and the importance of a secure connection.

**4. Is it well-structured?**
Yes, the answer is well-structured and easy to understand. It uses simple language and provides a clear explanation of the topic.

**Recommendations for improvement:**

* Provide specific examples of how TLS certificate validation can protect sensitive user data.
* Explain the concept of a trusted certificate authority (CA) and how it issues and verifies TLS certificates.
* Provide a brief overview of the TLS protocol and how it ensures the security of communication channels.

--- REVISED ANSWER ---
Sure, here is the revised answer:

TLS certificate validation is a vital security measure for mobile apps to ensure the authenticity and trustworthiness of the communication channels they use. It verifies the identity of the app and protects sensitive user data by ensuring that only authorized parties can access sensitive information. TLS certificate validation also helps establish a secure connection by verifying that the communication channel is encrypted and protected against eavesdropping.



"""