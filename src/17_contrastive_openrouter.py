"""
Title: Contrastive Prompting (OpenRouter)

Contrastive prompting shows the model correct examples AND common mistakes
with the right answer next to each one, so it learns what to avoid as well
as what to do. The same messages are classified with both styles to compare.

uv run python src/17_contrastive_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct", 
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Standard few-shot (good examples only)
    standard_few_shot = ChatPromptTemplate.from_template("""
        Classify each message as PHISHING or LEGITIMATE.

        GOOD EXAMPLES:
        Message: "Your account will be closed in 1 hour! Click here to verify your password."
        Label: PHISHING

        Message: "Your monthly statement is ready. Log in to the official app to view it."
        Label: LEGITIMATE

        Message: "Congratulations! You won ₹50 lakh. Share your bank details to claim."
        Label: PHISHING

        Now classify:
        Message: "{message}"
        Label:
    """)

    # Contrastive prompting (good AND bad examples)
    contrastive_prompt = ChatPromptTemplate.from_template("""
        Classify each message as PHISHING or LEGITIMATE.

        CORRECT CLASSIFICATIONS:
        Yes "Your account will be closed in 1 hour! Click here to verify your password." -> PHISHING
          (Reason: fear and a link that asks for a password)

        Yes "Your monthly statement is ready. Log in to the official app to view it." -> LEGITIMATE
          (Reason: asks for nothing, points to the official app)

        Yes "Congratulations! You won ₹50 lakh. Share your bank details to claim." -> PHISHING
          (Reason: a prize that asks for bank details)

        INCORRECT CLASSIFICATIONS (avoid these mistakes):
        No "Your OTP is 482913. Do not share it with anyone." -> PHISHING (WRONG!)
          (Correct: LEGITIMATE - a normal OTP message that asks for nothing)

        No "This is IT. Sign in with your email password to open the attached invoice." -> LEGITIMATE (WRONG!)
          (Correct: PHISHING - it asks for a password to open a file)

        No "Urgent: the team meeting moved to 3 PM today." -> PHISHING (WRONG!)
          (Correct: LEGITIMATE - urgent, but asks for no password, money or link)

        Now classify:
        Message: "{message}"
        Label (PHISHING or LEGITIMATE):
    """)

    # Test on tricky cases
    tricky_messages = [
        "Your card was used for ₹2,499 at an online store. If this was not you, call the number on the back of your card.",  # No link, safe action
        "Your parcel is held at customs. Pay ₹49 at the link below to release it today.",  # Small payment trick
        "Security alert: new sign-in from a new device. If this was you, no action is needed.",  # Asks for nothing
        "Your KYC has expired. Update now at bit.ly/kyc-upd8 or your account will be blocked.",  # Fear and a short link
    ]

    print("=== STANDARD FEW-SHOT ===\n")
    standard_chain = standard_few_shot | llm | parser
    for i, message in enumerate(tricky_messages, 1):
        result = standard_chain.invoke({"message": message})
        print(f'{i}. "{message}"\n  -> {result.strip()}\n')

    print("\n=== CONTRASTIVE PROMPTING ===\n")
    contrastive_chain = contrastive_prompt | llm | parser
    for i, message in enumerate(tricky_messages, 1):
        result = contrastive_chain.invoke({"message": message})
        print(f'{i}. "{message}"\n  -> {result.strip()}\n')


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

=== STANDARD FEW-SHOT ===

1. "Your card was used for ₹2,499 at an online store. If this was not you, call the number on the back of your card."
  -> LEGITIMATE

2. "Your parcel is held at customs. Pay ₹49 at the link below to release it today."
  -> PHISHING

3. "Security alert: new sign-in from a new device. If this was you, no action is needed."
  -> Label: LEGITIMATE

4. "Your KYC has expired. Update now at bit.ly/kyc-upd8 or your account will be blocked."
  -> PHISHING


=== CONTRASTIVE PROMPTING ===

1. "Your card was used for ₹2,499 at an online store. If this was not you, call the number on the back of your card."
  -> LEGITIMATE

(The message alerts about a transaction and asks you to call the number on the back of your card — it does not ask for any password, bank details, or sensitive information, and directs you to the legitimate customer service channel.)

2. "Your parcel is held at customs. Pay ₹49 at the link below to release it today."
  -> PHISHING
(Reason: creates urgency, asks for money, and includes a suspicious link)

3. "Security alert: new sign-in from a new device. If this was you, no action is needed."
  -> LEGITIMATE

(Reason: It is a standard security notification informing the user about a new sign-in. It does not ask for any password, personal information, or ask the user to click a link. The message reassures the user that no action is needed if they initiated the sign-in.)

4. "Your KYC has expired. Update now at bit.ly/kyc-upd8 or your account will be blocked."
  -> PHISHING
(Reason: fear of account being blocked combined with a suspicious shortened link that likely leads to a credential or personal data harvesting page)


"""