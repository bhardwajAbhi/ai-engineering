"""
Title: Contrastive Prompting (Google Generative AI)

Contrastive prompting shows the model correct examples AND common mistakes
with the right answer next to each one, so it learns what to avoid as well
as what to do. The same items are classified with both styles to compare.

uv run python src/17_contrastive_google.py
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
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Standard few-shot (good examples only)
    standard_few_shot = ChatPromptTemplate.from_template("""
        Classify each item as PERSONAL or NOT_PERSONAL.
        Rule: data is PERSONAL when it is about an identifiable individual,
        on its own or when linked to an account. Otherwise it is NOT_PERSONAL.

        GOOD EXAMPLES:
        Item: "Full name and mobile number of a customer"
        Label: PERSONAL

        Item: "Total number of app downloads this month"
        Label: NOT_PERSONAL

        Item: "Aadhaar number of a user"
        Label: PERSONAL

        Now classify:
        Item: "{item}"
        Label:
    """)

    # Contrastive prompting (good AND bad examples)
    contrastive_prompt = ChatPromptTemplate.from_template("""
        Classify each item as PERSONAL or NOT_PERSONAL.
        Rule: data is PERSONAL when it is about an identifiable individual,
        on its own or when linked to an account. Otherwise it is NOT_PERSONAL.

        CORRECT CLASSIFICATIONS:
        Yes "Full name and mobile number of a customer" -> PERSONAL
          (Reason: directly identifies one person)

        Yes "Total number of app downloads this month" -> NOT_PERSONAL
          (Reason: a count, not about any one person)

        Yes "Aadhaar number of a user" -> PERSONAL
          (Reason: a unique identity number of one person)

        INCORRECT CLASSIFICATIONS (avoid these mistakes):
        No "IP address of a user who is signed in" -> NOT_PERSONAL (WRONG!)
          (Correct: PERSONAL - linked to a signed-in account, it points to one person)

        No "Registration number of a company" -> PERSONAL (WRONG!)
          (Correct: NOT_PERSONAL - it is about a company, not an individual)

        No "Average age of all users is 29" -> PERSONAL (WRONG!)
          (Correct: NOT_PERSONAL - an average does not identify anyone)

        Now classify:
        Item: "{item}"
        Label (PERSONAL or NOT_PERSONAL):
    """)

    # Test on tricky cases
    tricky_items = [
        "Advertising ID of a phone that is linked to a user's account",  # Linked to a person
        "GST number of a registered private limited company",  # About a company
        "Average battery level across all user devices",  # Aggregate
        "Email address an employee uses to sign in to the app",  # Identifies a person
    ]

    print("=== STANDARD FEW-SHOT ===\n")
    standard_chain = standard_few_shot | llm | parser
    for i, item in enumerate(tricky_items, 1):
        result = standard_chain.invoke({"item": item})
        print(f'{i}. "{item}"\n  -> {result.strip()}\n')

    print("\n=== CONTRASTIVE PROMPTING ===\n")
    contrastive_chain = contrastive_prompt | llm | parser
    for i, item in enumerate(tricky_items, 1):
        result = contrastive_chain.invoke({"item": item})
        print(f'{i}. "{item}"\n  -> {result.strip()}\n')


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

=== STANDARD FEW-SHOT ===

1. "Advertising ID of a phone that is linked to a user's account"
  -> Label: PERSONAL

2. "GST number of a registered private limited company"
  -> Label: NOT_PERSONAL

3. "Average battery level across all user devices"
  -> Label: NOT_PERSONAL

4. "Email address an employee uses to sign in to the app"
  -> Label: PERSONAL


=== CONTRASTIVE PROMPTING ===

1. "Advertising ID of a phone that is linked to a user's account"
  -> Label (PERSONAL or NOT_PERSONAL): PERSONAL

2. "GST number of a registered private limited company"
  -> Label (PERSONAL or NOT_PERSONAL): NOT_PERSONAL

3. "Average battery level across all user devices"
  -> Label (PERSONAL or NOT_PERSONAL): NOT_PERSONAL

4. "Email address an employee uses to sign in to the app"
  -> Label (PERSONAL or NOT_PERSONAL): PERSONAL


"""