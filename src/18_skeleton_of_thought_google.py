"""
Title: Skeleton of Thought Prompting (Google Generative AI)

Skeleton of thought writes a short outline first, then expands each section
of the outline on its own, and finally joins the sections into one article.
Planning the structure first keeps long writing organized.

uv run python src/18_skeleton_of_thought_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def skeleton_of_thought(task: str, skeleton_chain, expand_chain, combine_chain) -> str:
    """Generate long-form content using the skeleton-first approach."""

    # Step 1: Generate the skeleton
    skeleton = skeleton_chain.invoke({"task": task})
    print("SKELETON:")
    print(skeleton)
    print("\n" + "=" * 60 + "\n")

    # Parse the sections from the skeleton (lines that start with a number)
    # Models often wrap titles in markdown like **1. Title**, so strip * and # first
    sections = []
    for line in skeleton.split("\n"):
        line = line.strip().strip("*# ")
        if line and line[0].isdigit():
            sections.append(line)

    # Step 2: Expand each section
    expanded = []
    for section in sections:
        print(f"Expanding: {section[:50]}...")
        content = expand_chain.invoke({"task": task, "skeleton": skeleton, "section": section})
        expanded.append(f"## {section}\n\n{content}")

    print("\n" + "=" * 60 + "\n")

    # Step 3: Combine the sections
    final = combine_chain.invoke({"task": task, "expanded_sections": "\n\n---\n\n".join(expanded)})

    return final


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Step 1 prompt: generate the skeleton
    skeleton_prompt = ChatPromptTemplate.from_template("""
        Create a structured outline for the following writing task.
        Include 4-6 main sections with brief descriptions of what each should cover.

        TASK: {task}

        OUTLINE:
        1. [Section Title]: [Brief description of content]
        2. [Section Title]: [Brief description of content]
        ...
    """)

    # Step 2 prompt: expand one section
    expand_prompt = ChatPromptTemplate.from_template("""
        Write the content for this section of an article.

        ARTICLE TOPIC: {task}

        FULL OUTLINE:
        {skeleton}

        SECTION TO EXPAND: {section}

        Write 2-3 paragraphs for this section only. Be specific and substantive.
    """)

    # Step 3 prompt: combine the sections
    combine_prompt = ChatPromptTemplate.from_template("""
        Combine these sections into a cohesive article with smooth transitions.

        TOPIC: {task}

        SECTIONS:
        {expanded_sections}

        Create the final article with an introduction and conclusion.
        Ensure smooth transitions between sections.
    """)

    # Create the three chains
    skeleton_chain = skeleton_prompt | llm | parser
    expand_chain = expand_prompt | llm | parser
    combine_chain = combine_prompt | llm | parser

    # A long-form writing task
    task = (
        "Write an article explaining how an Android app should ask for, record "
        "and honor user consent under India's DPDP Act, 2023, for a developer audience"
    )

    print(f"TASK: {task}")
    print("=" * 60 + "\n")

    article = skeleton_of_thought(task, skeleton_chain, expand_chain, combine_chain)
    print("FINAL ARTICLE:")
    print(article)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

TASK: Write an article explaining how an Android app should ask for, record and honor user consent under India's DPDP Act, 2023, for a developer audience
============================================================

SKELETON:
Here is a structured outline for an article tailored to Android developers regarding compliance with India’s Digital Personal Data Protection (DPDP) Act, 2023.

### **Outline: Implementing DPDP Act Compliance in Android Applications**

**1. Understanding the DPDP Act for Developers**
*   Provide a high-level overview of the DPDP Act, focusing on the shift from "implied" to "explicit" consent. Define key terms like *Data Fiduciary* (the app owner) and *Data Principal* (the user), and explain why technical implementation is now a legal requirement rather than just a UX choice.

**2. Designing Compliant Consent Requests (The "Notice" Requirement)**
*   Explain the legal requirement to provide a "Notice" before or at the time of data collection. Detail how to design UI/UX flows that are clear, concise, and available in multiple languages (as required by the Act). Emphasize that consent must be "free, specific, informed, unconditional, and unambiguous."

**3. Technical Implementation: Recording and Managing Consent**
*   Discuss the backend and local storage architecture required to maintain a "Consent Ledger." Explain how to programmatically link specific data points (e.g., location, contacts) to a timestamped, version-controlled consent record. Include best practices for handling consent withdrawal (the "Right to Withdraw") via app settings.

**4. Data Minimization and Purpose Limitation**
*   Explain the technical constraints developers must implement to ensure data is only used for the specific purpose for which it was collected. Discuss how to use Android’s scoped storage and permission models to enforce "purpose limitation" and ensure that data is deleted or anonymized once the purpose is fulfilled.

**5. Handling Data Principal Rights (Access, Correction, and Erasure)**
*   Detail the technical workflows required to fulfill user requests under the Act. This includes building features that allow users to download their data (portability), request corrections, and trigger a "Right to Erasure" (the "Right to be Forgotten") that propagates across your databases and third-party SDKs.

**6. Auditing, Logging, and Security Best Practices**
*   Focus on the "Accountability" pillar of the Act. Outline how to implement robust logging for consent changes, perform regular security audits, and ensure that data processing is encrypted both in transit and at rest to meet the Act’s security standards.

============================================================

Expanding: 1. Understanding the DPDP Act for Developers...
Expanding: 2. Designing Compliant Consent Requests (The "Noti...
Expanding: 3. Technical Implementation: Recording and Managin...
Expanding: 4. Data Minimization and Purpose Limitation...
Expanding: 5. Handling Data Principal Rights (Access, Correct...
Expanding: 6. Auditing, Logging, and Security Best Practices...

============================================================

FINAL ARTICLE:
# Implementing the DPDP Act: A Developer’s Guide to Consent and Compliance

The Digital Personal Data Protection (DPDP) Act, 2023, marks a paradigm shift in India’s digital landscape. For Android developers, this is more than a legal update; it is a fundamental architectural challenge. The Act moves the industry away from "implied" consent—often buried in opaque Terms of Service—toward a regime of explicit, granular, and verifiable authorization. As a **Data Fiduciary**, you are now responsible for ensuring that every byte of personal data collected is backed by a clear, legal mandate. This guide outlines how to transition your Android application from a "collect-all" mindset to a privacy-first, compliant architecture.

### 1. Understanding the DPDP Act for Developers
Under the DPDP Act, your app acts as the Data Fiduciary, while the user is the Data Principal. The Act mandates that consent must be free, specific, informed, unconditional, and unambiguous. For you, this means "I agree" checkboxes are no longer legally sufficient. Compliance is now a core architectural requirement, necessitating a move away from monolithic permission requests toward a modular, event-driven consent architecture. You are legally obligated to provide a "Notice" that informs the user of the data being collected and the purpose of processing, ideally in both English and any of the 22 languages specified in the Eighth Schedule of the Indian Constitution.

### 2. Designing Compliant Consent Requests (The "Notice" Requirement)
The "Notice" is a functional prerequisite for data collection. Before or at the time of collection, you must provide a clear, itemized notice detailing the personal data being collected and the specific purpose for that processing. To avoid "dark patterns," move away from "blanket" permissions. Instead, implement granular consent screens that map specific data points to their intended use cases (e.g., "We need your location to provide real-time delivery tracking").

Leverage Android’s `ActivityResultContracts` or custom bottom sheets to trigger *just-in-time* requests. By requesting permissions only when a user engages with a specific feature, you ensure consent is "informed and specific." Furthermore, because the Act requires consent to be "unconditional," you must ensure that services are not contingent on unnecessary data collection. If a user denies a non-essential permission, your app should gracefully degrade functionality rather than locking the user out.

### 3. Technical Implementation: Recording and Managing Consent
To prove compliance, you must move beyond simple boolean flags and implement a robust "Consent Ledger." This ledger should be an immutable, version-controlled audit trail that links every data point to a specific, timestamped consent event. Your backend should generate a cryptographically signed record capturing the user’s ID, the purpose of processing, the version of the privacy notice accepted, and the timestamp.

Managing the "Right to Withdraw" requires a centralized "Privacy Dashboard" within your app settings. When a user revokes consent, your system must trigger an event-driven workflow to propagate this change to all microservices and third-party SDKs. By treating consent as a dynamic, stateful object, you prevent "zombie" data processing and ensure your application remains compliant as user preferences evolve.

### 4. Data Minimization and Purpose Limitation
The principle of "purpose limitation" dictates that you may only process data for the specific purpose for which consent was granted. Architecturally, this requires "Data Tagging," where every data point is metadata-linked to a `consent_id` and a `purpose_code`. Enforce these constraints at the API layer to ensure that location data collected for "Delivery" cannot be repurposed for "Ad Targeting."

Furthermore, leverage Android’s Scoped Storage and implement automated data lifecycle policies—such as Time-To-Live (TTL) indices—to ensure that once a purpose is fulfilled, the data is purged or anonymized. By treating data as a liability that expires rather than an asset to be hoarded, you align your infrastructure with the Act’s mandate to retain data only as long as necessary.

### 5. Handling Data Principal Rights (Access, Correction, and Erasure)
The DPDP Act grants users the right to access, correct, and erase their data. This requires a "Request Orchestration" workflow. When a user requests their data, your system must aggregate information from primary databases, caches, logs, and third-party SDKs, providing it in a machine-readable format like JSON.

The "Right to Erasure" requires a cascading deletion strategy. When a user triggers an erasure request, your backend must propagate this command across your entire ecosystem, including third-party partners. Use "Data Deletion Webhooks" to notify external SDKs that the user has withdrawn consent, ensuring their data is purged across the entire supply chain.

### 6. Auditing, Logging, and Security Best Practices
Finally, the "Accountability" pillar mandates that you maintain a verifiable trail of compliance. Your Consent Audit Log should be stored in a write-once-read-many (WORM) environment to prevent tampering. Beyond logging, enforce rigorous security standards: use TLS 1.3 for data in transit and AES-256 for data at rest. 

Because you are responsible for the actions of your data processors, conduct regular security audits of all third-party SDKs. Use automated dependency scanning and strict `network_security_config.xml` configurations to minimize the risk of unauthorized data exposure.

### Conclusion
Compliance with the DPDP Act is not a one-time legal hurdle; it is a continuous technical commitment. By building your Android application with granular consent, immutable audit logs, and automated data lifecycle management, you not only satisfy the Data Protection Board of India but also build deeper trust with your users. In the modern digital economy, privacy is a feature—and for the compliant developer, it is a competitive advantage.

"""