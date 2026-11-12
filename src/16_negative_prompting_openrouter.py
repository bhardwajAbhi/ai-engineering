"""
Title: Negative Prompting (OpenRouter)

Negative prompting tells the model what NOT to do as well as what to do,
by adding clear limits such as a word limit and a list of things to avoid.
The same request is run with and without these limits so you can compare.

uv run python src/16_negative_prompting_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        #model="qwen/qwen-2.5-7b-instruct",
        model="meta-llama/llama-3.1-8b-instruct",
        temperature=0.3,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    print("=" * 60)
    print("NEGATIVE PROMPTING: WITH AND WITHOUT CONSTRAINTS")
    print("=" * 60)

    topic = "certificate pinning in mobile apps"

    # Without constraints
    unconstrained_prompt = ChatPromptTemplate.from_template(
        "Explain {topic} to a non-technical product manager."
    )

    # With negative constraints
    constrained_prompt = ChatPromptTemplate.from_template("""
        Explain {topic} to a non-technical product manager.

        REQUIREMENTS:
        - Maximum 80 words
        - Use an everyday analogy, not a technical one
        - Focus on what it protects and why it matters to users

        DO NOT:
        - Use technical jargon (no "man-in-the-middle", "public key", "TLS", "handshake")
        - Include code examples
        - Use bullet points
        - Mention specific libraries or tools

        Write in a conversational, professional tone.
    """)

    print("\n--- WITHOUT CONSTRAINTS ---")
    result = (unconstrained_prompt | llm | parser).invoke({"topic": topic})
    print(result)
    word_count = len(result.split())
    print(f"\n[Word count: {word_count}]")

    print("\n" + "=" * 60)

    print("\n--- WITH CONSTRAINTS ---")
    result = (constrained_prompt | llm | parser).invoke({"topic": topic})
    print(result)
    word_count = len(result.split())
    print(f"\n[Word count: {word_count}]")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

============================================================
NEGATIVE PROMPTING: WITH AND WITHOUT CONSTRAINTS
============================================================

--- WITHOUT CONSTRAINTS ---
Certificate pinning is a security feature in mobile apps that helps prevent malicious activities, such as man-in-the-middle (MITM) attacks. I'll explain it in a way that's easy to understand, even if you're not a technical expert.

**What is a certificate?**

In the context of mobile apps, a certificate is like a digital ID that proves the app's authenticity. It's issued by a trusted authority, like a certificate authority (CA), and contains information about the app's identity, such as its name, version, and other details.

**What is a MITM attack?**

Imagine you're trying to access a secure website, but your internet connection is being intercepted by a malicious person. They can then see and modify the data you're sending, which is a big security risk. A MITM attack is like this, but on a mobile app. An attacker can intercept the communication between your app and the server, allowing them to steal sensitive data or inject malware.

**How does certificate pinning prevent MITM attacks?**

Certificate pinning is a technique that ensures the app only communicates with the server using a specific, trusted certificate. Here's how it works:

1. The app is configured to only accept connections from a specific certificate, which is "pinned" to the app.
2. When the app tries to connect to the server, it checks the server's certificate against the pinned certificate.
3. If the server's certificate matches the pinned certificate, the app allows the connection to proceed.
4. If the server's certificate doesn't match the pinned certificate, the app refuses the connection and displays an error message to the user.

**Why is certificate pinning important?**

Certificate pinning is essential because it prevents an attacker from intercepting the communication between the app and the server. Even if an attacker manages to obtain a valid certificate, the app will still refuse the connection because the certificate doesn't match the pinned one.

**Example scenario**

Let's say you're a product manager for a banking app. You want to ensure that the app only communicates with the bank's server using a specific, trusted certificate. You configure the app to pin this certificate, so that even if an attacker tries to intercept the communication, the app will still refuse the connection and display an error message to the user.

In summary, certificate pinning is a security feature that helps prevent MITM attacks by ensuring that the app only communicates with the server using a specific, trusted certificate. It's an essential security measure for mobile apps that handle sensitive data or require high security.

[Word count: 420]

============================================================

--- WITH CONSTRAINTS ---
Here's an explanation of certificate pinning in mobile apps that a non-technical product manager can understand:

Imagine you're meeting a friend at a coffee shop. You've met before, so you know what they look like. If someone else showed up claiming to be your friend, you'd be suspicious. Certificate pinning is like verifying the identity of a website or app by checking its digital "face" or signature, ensuring it's the real deal and not an imposter. This protects users from fake or malicious sites that try to impersonate trusted ones.

[Word count: 90]

"""