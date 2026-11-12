"""
Title: Negative Prompting (Google Generative AI)

Negative prompting tells the model what NOT to do as well as what to do,
by adding clear limits such as a word limit and a list of things to avoid.
The same request is run with and without these limits so you can compare.

uv run python src/16_negative_prompting_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.3,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    print("=" * 60)
    print("NEGATIVE PROMPTING: WITH AND WITHOUT CONSTRAINTS")
    print("=" * 60)

    topic = "India's DPDP Act, 2023 and consent for mobile apps"

    # Without constraints
    unconstrained_prompt = ChatPromptTemplate.from_template(
        "Explain {topic} to a startup founder."
    )

    # With negative constraints
    constrained_prompt = ChatPromptTemplate.from_template("""
        Explain {topic} to a startup founder.

        REQUIREMENTS:
        - Maximum 100 words
        - Use everyday business analogies, not legal ones
        - Focus on what the founder must do and the cost of getting it wrong

        DO NOT:
        - Use legal jargon (no "data fiduciary", "data principal", "Schedule", "Section")
        - Include code or technical terms
        - Use bullet points
        - Quote exact penalty amounts

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
Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.

For a startup founder, the **Digital Personal Data Protection (DPDP) Act, 2023** is not just a legal hurdle—it is a fundamental shift in how you must design your product. 

Think of it as the "GDPR of India." If you handle user data, you are now a **Data Fiduciary**, and your users are **Data Principals**.

Here is the breakdown of what you need to know, specifically regarding **Consent** for your mobile app.

---

### 1. The Core Philosophy: "Privacy by Design"
The Act mandates that you can only process data for a **specific, lawful purpose** that the user has clearly consented to. You can no longer collect "everything just in case we need it later."

### 2. The "Consent" Requirements (The Rules of the Game)
Under the DPDP Act, consent must be **free, specific, informed, unconditional, and unambiguous.**

*   **The "Notice" Requirement:** Before or at the time of asking for consent, you must provide a **Notice** (in plain language, and ideally in English + the 22 languages of the 8th Schedule of the Constitution). This notice must state:
    *   What data you are collecting.
    *   Why you are collecting it (the purpose).
    *   How they can withdraw consent.
    *   How they can file a grievance with your Data Protection Officer (DPO).
*   **Actionable Consent:** You cannot use "pre-ticked boxes" or "silence" as consent. The user must perform a clear, affirmative action (e.g., clicking "I Agree" after reading the notice).
*   **Withdrawal:** It must be **as easy to withdraw consent as it was to give it.** If a user clicks "Withdraw," you must stop processing their data and delete it (unless another law requires you to keep it).

### 3. What this means for your Mobile App UX
Stop using "Dark Patterns." The Act explicitly discourages deceptive design.

*   **Granular Consent:** Don’t bundle permissions. If your app needs Location for a delivery feature but also wants to track Contacts for "marketing," you must ask for these separately. Do not force the user to accept both to use the app.
*   **The "Consent Manager":** The Act introduces a "Consent Manager"—a platform or entity registered with the Data Protection Board that acts as a single point for users to manage their consents across different apps. You will eventually need to integrate with these systems.
*   **Data Minimization:** Only collect what is strictly necessary for the service. If you are a food delivery app, you don't need access to the user's photo gallery. If you ask for it, you will have to justify it legally.

### 4. Critical "Startup" Checklist
1.  **Appoint a DPO:** If you are a significant data fiduciary (or handle large volumes of data), you must appoint a Data Protection Officer based in India.
2.  **Update your Privacy Policy:** Your current "Terms & Conditions" are likely insufficient. Rewrite them to be transparent, simple, and specific to the DPDP Act.
3.  **Data Audit:** Map your data flow. Where is it stored? Who has access? How long do you keep it? If you don't know where your data is, you cannot comply.
4.  **Children’s Data:** **Huge Red Flag.** You cannot track, target, or process the data of anyone under 18 in a way that causes "detrimental effect." You must obtain verifiable parental consent. If your app is for kids, the compliance bar is significantly higher.
5.  **Grievance Redressal:** You must have a clear mechanism for users to complain. If a user asks to delete their data, you must have a technical process to wipe it from your servers and your third-party vendors (e.g., AWS, Firebase, Mixpanel).

### 5. The Risk: Why you should care
The penalties under the DPDP Act are massive—up to **₹250 Crore** for failing to prevent a data breach. Beyond the fines, the reputational damage and the potential for the Data Protection Board to order you to stop processing data (effectively shutting down your app) are existential threats to a startup.

### Founder’s Advice:
*   **Don't treat this as a "legal" task; treat it as a "product" task.** Build the consent flow into your onboarding UX.
*   **Talk to your Devs:** Ensure your database architecture allows for "Right to Erasure" (deleting a user's data upon request).
*   **Consult a specialist:** This is not a DIY job. Spend the money to have a legal expert review your data flow once before you scale.

***Disclaimer:** I am an AI, not a lawyer. The DPDP Act is complex and subject to upcoming rules. Please consult with a legal professional specializing in Indian tech law before finalizing your compliance strategy.*

[Word count: 755]

============================================================

--- WITH CONSTRAINTS ---
Think of the DPDP Act as a "digital handshake" agreement. You are now legally required to ask users for clear, specific permission before collecting their personal info, just like asking for a guest’s coat before hanging it up. You must explain exactly why you need their data and let them take it back whenever they want. If you treat user data like a borrowed item rather than your own property, you’ll stay safe. If you ignore these rules, you risk massive fines that could bankrupt your startup and destroy the trust you’ve worked so hard to build.

[Word count: 97]

"""