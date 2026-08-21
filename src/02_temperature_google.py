"""
Title: Temperature Control in LLMs (Google Generative AI)

uv run python src/02_temperature_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    load_dotenv()

    prompt="Write a one sentense security awareness message about phishing emails."

    print("="*60)
    print("TEMPERATURE COMPARISION")
    print("="*60)

    # Temperature = 0: Low randomness and generally considered consistent output
    print("\n--- Temperature 0 (Low Randomness) ---")
    llm_zero = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
        max_output_tokens=64,
    )

    for i in range(3):
        response = llm_zero.invoke(prompt)
        print(f"Run {i+1}: {response.text}")



    # Temperature = 0.7: Balanced creativity
    print("\n--- Temperature 0.7 (Balanced Creativity) ---")
    llm_balanced = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.7,
        max_output_tokens=64,
    )

    for i in range(3):
        response = llm_balanced.invoke(prompt)
        print(f"Run {i+1}: {response.text}")
    


    # Temperature = 1.5: High creativity
    print("\n--- Temperature 1.5 (High Creativity) ---")
    llm_creative = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=1.5,
        max_output_tokens=64,
    )

    for i in range(3):
        response = llm_creative.invoke(prompt)
        print(f"Run {i+1}: {response.text}")



if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
TEMPERATURE COMPARISION
============================================================

--- Temperature 0 (Low Randomness) ---
Run 1: Always pause and verify the sender's address before clicking any links or downloading attachments, as phishing emails are designed to trick you into compromising your security.
Run 2: Always pause and verify the sender's address before clicking any links or downloading attachments, as phishing emails are designed to trick you into compromising your security.
Run 3: Always pause and verify the sender's address before clicking any links or downloading attachments, as phishing emails are designed to trick you into compromising your security.

--- Temperature 0.7 (Balanced Creativity) ---
Run 1: Always pause and verify the sender's identity before clicking any links or downloading attachments, as phishing emails are designed to trick you into compromising your security.
Run 2: Always verify the sender's address and think twice before clicking any links or downloading attachments, as phishing emails are designed to trick you into compromising your security.
Run 3: Always pause and verify the sender's address before clicking any links or downloading attachments, as phishing emails are designed to look legitimate while stealing your sensitive information.

--- Temperature 1.5 (High Creativity) ---
Run 1: Always scrutinize the sender's address and think twice before clicking any links or downloading attachments, as even familiar-looking emails can be sophisticated phishing attempts designed to steal your credentials.
Run 2: Always verify the sender's address and think twice before clicking any links or attachments, as phishers often disguise their emails to steal your sensitive information.
Run 3: Always verify the sender's address and think twice before clicking any links or attachments, as phishing emails are designed to look legitimate to steal your personal information.


"""