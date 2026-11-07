"""
Title: Chain of Verification Prompting (OpenRouter)
Chain of Verification asks the model to write an answer, pull out the
facts it stated, check each fact on its own and then fix any fact found
to be wrong. Checking small facts one at a time is easier than judging
a whole answer at once.
uv run python src/14_chain_of_verification_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
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
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
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
        "Tell me about the Heartbleed vulnerability in OpenSSL: its CVE "
        "number, when it was disclosed, which versions were affected, how "
        "the bug worked and how much data it could leak."
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

Tell me about the Heartbleed vulnerability in OpenSSL: its CVE number, when it was disclosed, which versions were affected, how the bug worked and how much data it could leak.

============================================================

INITIAL RESPONSE:
The Heartbleed vulnerability is a critical security flaw in the OpenSSL cryptographic software library, which is widely used to secure internet communications. Here are the specific details you requested:

### CVE Number and Disclosure Date
- **CVE Number:** The Heartbleed bug was assigned the CVE identifier [CVE-2014-0160](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2014-0160).
- **Disclosure Date:** The vulnerability was publicly disclosed on April 7, 2014.

### Affected Versions
- **Versions Affected:** The Heartbleed bug affected all versions of OpenSSL prior to version 1.0.1g, specifically versions 1.0.1 through 1.0.1f were vulnerable. Version 1.0.1g and later are not affected by this issue.

### How the Bug Worked
The Heartbleed bug exploited a flaw in the implementation of the TLS heartbeat extension in OpenSSL. This extension allows one end of a TLS connection to send a "heartbeat" message to the other end to check if the connection is still alive. The flaw occurred because OpenSSL did not properly validate the length of the data sent in the heartbeat response. An attacker could send a specially crafted heartbeat request that would cause the server to return more data than it should have, potentially including sensitive information such as private keys, usernames, passwords, and other confidential data.

### Data Leak Potential
- **Amount of Data Leaked:** The amount of data that could be leaked varied depending on the implementation and configuration of the affected server. In the worst-case scenario, an attacker could potentially leak up to 64 kilobytes of memory at a time from the server's memory. This could include:
  - Private encryption keys (which could allow the attacker to decrypt or forge encrypted communications)
  - Usernames and passwords
  - Sensitive application data
  - Internal network addresses and configurations

### Impact
The Heartbleed bug was considered one of the most serious security vulnerabilities ever discovered due to its widespread impact and the potential for catastrophic data breaches. It affected a large number of websites and services, leading to significant security concerns and the need for widespread patching and remediation efforts.

### Mitigation
To mitigate the risk posed by Heartbleed, organizations needed to:
- Update their OpenSSL installations to version 1.0.1g or later.
- Regenerate and redistribute SSL/TLS certificates.
- Perform a thorough review of their systems to identify and address any potential exposure of sensitive data.

In summary, the Heartbleed vulnerability was a critical flaw in OpenSSL that allowed attackers to potentially steal sensitive data from servers using the affected versions of OpenSSL.

----------------------------------------
EXTRACTED CLAIMS:
  1. The Heartbleed vulnerability is a critical security flaw in the OpenSSL cryptographic software library.
  2. The Heartbleed bug was assigned the CVE identifier [CVE-2014-0160].
  3. The vulnerability was publicly disclosed on April 7, 2014.
  4. The Heartbleed bug affected all versions of OpenSSL prior to version 1.0.1g.
  5. Versions 1.0.1 through 1.0.1f were specifically vulnerable to the Heartbleed bug.
  6. An attacker could send a specially crafted heartbeat request that would cause the server to return more data than it should have.
  7. In the worst-case scenario, an attacker could potentially leak up to 64 kilobytes of memory at a time from the server's memory.
  8. The amount of data that could be leaked varied depending on the implementation and configuration of the affected server.
  9. The Heartbleed bug allowed for the potential theft of private encryption keys, usernames and passwords, sensitive application data, and internal network addresses and configurations.
  10. The Heartbleed bug was considered one of the most serious security vulnerabilities ever discovered due to its widespread impact and potential for catastrophic data breaches.
  11. Organizations needed to update their OpenSSL installations to version 1.0.1g or later to mitigate the risk posed by Heartbleed.
  12. Organizations needed to regenerate and redistribute SSL/TLS certificates to mitigate the risk posed by Heartbleed.
  13. Organizations needed to perform a thorough review of their systems to identify and address any potential exposure of sensitive data.

----------------------------------------
VERIFICATION:
  1. The Heartbleed vulnerability is a critical security flaw in the OpenSSL cryptographic software library.
    -> VERIFIED: The Heartbleed vulnerability is indeed a critical security flaw in the OpenSSL cryptographic software library. It was discovered in 2014 and affected versions of OpenSSL that were using the TLS heartbeat extension improperly, allowing an attacker to potentially extract sensitive information from the server's memory.
  2. The Heartbleed bug was assigned the CVE identifier [CVE-2014-0160].
    -> VERIFIED: The Heartbleed bug was assigned the CVE identifier [CVE-2014-0160].
  3. The vulnerability was publicly disclosed on April 7, 2014.
    -> UNCERTAIN
  4. The Heartbleed bug affected all versions of OpenSSL prior to version 1.0.1g.
    -> INCORRECT: The Heartbleed bug affected all versions of OpenSSL prior to version 1.0.1f, not 1.0.1g.
  5. Versions 1.0.1 through 1.0.1f were specifically vulnerable to the Heartbleed bug.
    -> UNCERTAIN
  6. An attacker could send a specially crafted heartbeat request that would cause the server to return more data than it should have.
    -> VERIFIED: The claim is correct. In a heartbeat protocol, if not properly implemented, an attacker could send a specially crafted heartbeat request that causes the server to return more data than intended, potentially leading to information leakage or other security vulnerabilities.
  7. In the worst-case scenario, an attacker could potentially leak up to 64 kilobytes of memory at a time from the server's memory.
    -> UNCERTAIN
  8. The amount of data that could be leaked varied depending on the implementation and configuration of the affected server.
    -> VERIFIED: The amount of data that could be leaked indeed varied depending on the implementation and configuration of the affected server. Factors such as the type of data stored, access controls, encryption methods, and security protocols in place can all influence the extent of potential data leakage.
  9. The Heartbleed bug allowed for the potential theft of private encryption keys, usernames and passwords, sensitive application data, and internal network addresses and configurations.
    -> VERIFIED: The Heartbleed bug, which affected the OpenSSL cryptographic software library, indeed allowed for the potential theft of private encryption keys, usernames and passwords, sensitive application data, and internal network addresses and configurations.
  10. The Heartbleed bug was considered one of the most serious security vulnerabilities ever discovered due to its widespread impact and potential for catastrophic data breaches.
    -> VERIFIED: The Heartbleed bug, discovered in 2014, was indeed considered one of the most serious security vulnerabilities ever discovered. It affected implementations of the Transport Layer Security (TLS) protocol, specifically the TLS 1.2 implementation in the OpenSSL library. The vulnerability allowed attackers to potentially extract sensitive information from server memory, including private keys, usernames, and passwords, leading to significant security risks and data breaches.
  11. Organizations needed to update their OpenSSL installations to version 1.0.1g or later to mitigate the risk posed by Heartbleed.
    -> VERIFIED: The claim is accurate. Organizations needed to update their OpenSSL installations to version 1.0.1g or later to mitigate the risk posed by the Heartbleed bug, which was present in versions 1.0.1 through 1.0.1f.
  12. Organizations needed to regenerate and redistribute SSL/TLS certificates to mitigate the risk posed by Heartbleed.
    -> VERIFIED: Organizations needed to regenerate and redistribute SSL/TLS certificates to mitigate the risk posed by Heartbleed. The Heartbleed bug, discovered in 2014, affected systems using OpenSSL and could allow an attacker to steal sensitive information such as private keys used for SSL/TLS encryption. Regenerating and redistributing certificates was a critical step in addressing this vulnerability.
  13. Organizations needed to perform a thorough review of their systems to identify and address any potential exposure of sensitive data.
    -> VERIFIED: Organizations often need to perform a thorough review of their systems to identify and address any potential exposure of sensitive data to ensure compliance with regulations and protect against data breaches.

----------------------------------------
REVISED RESPONSE:
The Heartbleed vulnerability is a critical security flaw in the OpenSSL cryptographic software library, which is widely used to secure internet communications. Here are the specific details you requested:

### CVE Number and Disclosure Date
- **CVE Number:** The Heartbleed bug was assigned the CVE identifier [CVE-2014-0160](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2014-0160).
- **Disclosure Date:** The vulnerability was publicly disclosed on April 7, 2014.

### Affected Versions
- **Versions Affected:** The Heartbleed bug affected all versions of OpenSSL prior to version 1.0.1g, specifically versions 1.0.1 through 1.0.1f were vulnerable. Version 1.0.1g and later are not affected by this issue.

### How the Bug Worked
The Heartbleed bug exploited a flaw in the implementation of the TLS heartbeat extension in OpenSSL. This extension allows one end of a TLS connection to send a "heartbeat" message to the other end to check if the connection is still alive. The flaw occurred because OpenSSL did not properly validate the length of the data sent in the heartbeat response. An attacker could send a specially crafted heartbeat request that would cause the server to return more data than it should have, potentially including sensitive information such as private keys, usernames, passwords, and other confidential data.

### Data Leak Potential
- **Amount of Data Leaked:** The amount of data that could be leaked varied depending on the implementation and configuration of the affected server. In the worst-case scenario, an attacker could potentially leak up to 64 kilobytes of memory at a time from the server's memory. This could include:
  - Private encryption keys (which could allow the attacker to decrypt or forge encrypted communications)
  - Usernames and passwords
  - Sensitive application data
  - Internal network addresses and configurations

### Impact
The Heartbleed bug was considered one of the most serious security vulnerabilities ever discovered due to its widespread impact and the potential for catastrophic data breaches. It affected a large number of websites and services, leading to significant security concerns and the need for widespread patching and remediation efforts.

### Mitigation
To mitigate the risk posed by Heartbleed, organizations needed to:
- Update their OpenSSL installations to version 1.0.1g or later.
- Regenerate and redistribute SSL/TLS certificates.
- Perform a thorough review of their systems to identify and address any potential exposure of sensitive data.

In summary, the Heartbleed vulnerability was a critical flaw in OpenSSL that allowed attackers to potentially steal sensitive data from servers using the affected versions of OpenSSL.

### Verification Results Summary
1. **Critical Security Flaw:** VERIFIED
2. **CVE Identifier:** VERIFIED
3. **Disclosure Date:** UNCERTAIN (Corrected: April 7, 2014)
4. **Affected Versions:** INCORRECT (Corrected: All versions prior to 1.0.1g, specifically 1.0.1 through 1.0.1f)
5. **Heartbeat Request:** VERIFIED
6. **Worst-Case Scenario:** UNCERTAIN (Corrected: Up to 64 kilobytes of memory at a time)
7. **Data Leakage Variability:** VERIFIED
8. **Sensitive Data Theft:** VERIFIED
9. **Serious Vulnerability:** VERIFIED
10. **Mitigation Steps:** VERIFIED


"""