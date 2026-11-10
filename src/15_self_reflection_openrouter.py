"""
Title: Self-Reflection Prompting (OpenRouter)

Self-reflection asks the model to write a draft answer, then read that
draft as a reviewer and list specific improvements, and finally rewrite
the answer using that critique. Naming what is wrong first gives the
rewrite something concrete to fix.

uv run python src/15_self_reflection_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    # Temperature 0 keeps the draft, critique and revision steady.
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
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

    # Sample questions on app security and cryptography, each with a length limit
    questions = [
        "Explain what salting does for password hashes in 2 sentences.",
        "Explain why an Android app should not store an access token in plain text in SharedPreferences, in 3 sentences.",
        "Explain what certificate pinning is and why apps use it, in 3 sentences.",
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
QUESTION 1: Explain what salting does for password hashes in 2 sentences.
============================================================

--- DRAFT ---
Salting involves adding a unique random value to each password before hashing, which makes it harder for attackers to use precomputed hash tables. This process ensures that even if two users have the same password, their resulting hashes will be different.

--- CRITIQUE ---
### Critique

**1. Factual Correctness:**
- The answer is factually correct. It accurately describes the purpose of salting in password hashing, which is to prevent attackers from using precomputed hash tables (rainbow tables) and to ensure that identical passwords produce different hashes.

**2. Clarity and Conciseness:**
- The answer is clear and concise, effectively conveying the key points about salting.
- However, it could be slightly more concise by removing unnecessary words without losing clarity.

**3. Completeness:**
- The answer covers important aspects such as preventing the use of rainbow tables and ensuring unique hashes for identical passwords.
- It could benefit from mentioning another significant aspect of salting, such as increasing the complexity of brute-force attacks.

**4. Structure:**
- The structure is logical and easy to follow.
- The sentences are well-structured but could be improved for flow.

### Specific, Actionable Improvements

1. **Factual Correctness:**
   - No changes needed; the information provided is accurate.

2. **Clarity and Conciseness:**
   - Slightly refine the wording to make it even more concise:
     - Original: "This process ensures that even if two users have the same password, their resulting hashes will be different."
     - Improved: "This prevents identical passwords from producing the same hash."

2. **Completeness:**
   - Add a brief mention of how salting increases the difficulty of brute-force attacks:
     - Improved: "Salting involves adding a unique random value to each password before hashing, which makes it harder for attackers to use precomputed hash tables and increases the complexity of brute-force attacks. This ensures that even if two users have the same password, their resulting hashes will be different."

3. **Structure:**
   - Ensure smooth transitions between ideas:
     - Improved: "Salting involves adding a unique random value to each password before hashing, making it harder for attackers to use precomputed hash tables. This ensures that even if two users have the same password, their resulting hashes will be different, thereby increasing the complexity of brute-force attacks."

### Final Revised Answer

"Salting involves adding a unique random value to each password before hashing, making it harder for attackers to use precomputed hash tables. This ensures that even if two users have the same password, their resulting hashes will be different, thereby increasing the complexity of brute-force attacks."

--- REVISED ANSWER ---
Salting involves adding a unique random value to each password before hashing, making it harder for attackers to use precomputed hash tables and increasing the complexity of brute-force attacks. This ensures that even if two users have the same password, their resulting hashes will be different.

============================================================
QUESTION 2: Explain why an Android app should not store an access token in plain text in SharedPreferences, in 3 sentences.
============================================================

--- DRAFT ---
Storing an access token in plain text in SharedPreferences on Android makes it vulnerable to unauthorized access, as SharedPreferences data is stored as plain XML files that can be read by anyone with appropriate permissions. Additionally, this practice does not comply with security best practices for handling sensitive information, increasing the risk of data breaches and unauthorized API access.

--- CRITIQUE ---
### Critique

**1. Factual Correctness:**
- The answer is factually correct in stating that storing an access token in plain text in SharedPreferences makes it vulnerable to unauthorized access.
- However, the statement about SharedPreferences data being stored as plain XML files is not entirely accurate. SharedPreferences data is actually stored in a private file on the device, which is encrypted by default.

**2. Clarity and Conciseness:**
- The answer is clear and concise but could be more precise.
- It mentions security best practices but does not explicitly state what those best practices are.

**3. Completeness:**
- The answer covers the main points of vulnerability and non-compliance with security best practices.
- However, it could benefit from mentioning specific alternatives or better practices for storing access tokens securely.

**4. Structure:**
- The structure is logical and easy to follow.
- The sentences flow well, but there is room for improvement in sentence construction and flow.

### Specific, Actionable Improvements

1. **Factual Correctness:**
   - Replace "SharedPreferences data is stored as plain XML files" with "SharedPreferences data is stored in a private file on the device."
   - Add: "This file is encrypted by default, but it can still be accessed if the device is rooted or if the app has been compromised."

2. **Clarity and Conciseness:**
   - Enhance clarity by specifying what security best practices are violated.
   - Example: "Storing an access token in plain text in SharedPreferences violates security best practices such as encrypting sensitive information and using secure storage mechanisms."

3. **Completeness:**
   - Mention specific alternatives for secure storage.
   - Example: "Instead, consider using Android Keystore System or KeyStore to securely store access tokens."

4. **Structure:**
   - Improve sentence flow for better readability.
   - Example:
     - "Storing an access token in plain text in SharedPreferences makes it vulnerable to unauthorized access because the data is stored in a private file on the device, which can be accessed if the device is rooted or compromised. This practice also violates security best practices, such as encrypting sensitive information and using secure storage mechanisms. Instead, consider using Android Keystore System or KeyStore to securely store access tokens."

### Revised Answer

"Storing an access token in plain text in SharedPreferences makes it vulnerable to unauthorized access because the data is stored in a private file on the device, which can be accessed if the device is rooted or compromised. This practice also violates security best practices, such as encrypting sensitive information and using secure storage mechanisms. Instead, consider using Android Keystore System or KeyStore to securely store access tokens."

--- REVISED ANSWER ---
Storing an access token in plain text in SharedPreferences makes it vulnerable to unauthorized access because the data is stored in a private file on the device, which can be accessed if the device is rooted or compromised. This practice violates security best practices by failing to encrypt sensitive information and use secure storage mechanisms. Instead, consider using Android Keystore System or KeyStore for secure token storage.

============================================================
QUESTION 3: Explain what certificate pinning is and why apps use it, in 3 sentences.
============================================================

--- DRAFT ---
Certificate pinning is a security measure that binds a client application to the public key or certificate fingerprint of the server it expects to communicate with, ensuring that only trusted certificates are accepted. Apps use this technique to prevent man-in-the-middle attacks by verifying the server's certificate against a hardcoded list of acceptable certificates, thereby enhancing security.

--- CRITIQUE ---
### Critique

1. **Factual Correctness:**
   - The answer is factually correct and provides an accurate explanation of certificate pinning.

2. **Clarity and Conciseness:**
   - The answer is clear and concise, effectively conveying the concept in three sentences.
   - However, it could be slightly more explicit about the benefits and use cases to enhance clarity.

3. **Completeness:**
   - The answer covers the core aspects of what certificate pinning is and why apps use it.
   - It mentions man-in-the-middle attacks but does not explicitly state that certificate pinning helps prevent such attacks.

4. **Structure:**
   - The structure is good, with a clear introduction, explanation, and conclusion.
   - The sentences flow well, but adding a bit more detail could make it even clearer.

### Specific, Actionable Improvements

1. **Enhance Clarity on Benefits:**
   - Explicitly mention that certificate pinning prevents man-in-the-middle attacks by ensuring the server's certificate matches the expected one.
   
2. **Add a Brief Mention of Use Cases:**
   - Include a brief mention of common scenarios where certificate pinning is used, such as securing mobile applications or web services.

3. **Refine Sentence Structure for Flow:**
   - Slightly rephrase the sentences to ensure smooth flow and readability.

### Revised Answer

"Certificate pinning is a security measure that binds a client application to the public key or certificate fingerprint of the server it expects to communicate with, ensuring that only trusted certificates are accepted. This technique prevents man-in-the-middle attacks by verifying the server's certificate against a hardcoded list of acceptable certificates, thereby enhancing security. Certificate pinning is commonly used in mobile applications and web services to protect against unauthorized access."

This revised version is more detailed while maintaining clarity and conciseness.

--- REVISED ANSWER ---
"Certificate pinning is a security measure that binds a client application to the public key or certificate fingerprint of the server it expects to communicate with, ensuring only trusted certificates are accepted. This technique prevents man-in-the-middle attacks by verifying the server's certificate against a hardcoded list of acceptable certificates, thereby enhancing security. Certificate pinning is commonly used in mobile applications and web services to protect against unauthorized access."


"""