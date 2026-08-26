"""
Title: Zero Shot Prompting (Google Generative AI)
uv run python src/04_zero_shot_google.py
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

    # Parser to parse the LLM output
    parser = StrOutputParser()

    print("=" * 80)
    print("ZERO-SHOT INDIAN FOOD REVIEW SENTIMENT ANALYSIS")
    print("=" * 80)

    # Zero-shot food review sentiment classification prompt
    sentiment_prompt = ChatPromptTemplate.from_template("""
        Classify the sentiment of the following Indian food review as 
        POSITIVE, NEGATIVE, or NEUTRAL.
        
        Respond with only the classification label.

        Food Review: {review}

        Sentiment:""")

    # The Chain
    chain = sentiment_prompt | llm | parser

    # Sample Indian food reviews
    sample_reviews = [
        "The Punjabi Amritsari kulcha was crispy, buttery, and absolutely delicious.",
        "The Tamil Nadu masala dosa was cold, soggy, and disappointing.",
        "The Bengali mishti doi was served after the main meal.",
        "The Gujarati dhokla was fine, but there was nothing special about it.",
        "The Hyderabadi dum biryani was aromatic, flavourful, and perfectly spiced.",
    ]
    
    # Invoking the Chain for each review and classifying
    for review in sample_reviews:
        result = chain.invoke({"review": review})
        print(f"\nFood Review: {review}")
        print(f"Sentiment: {result.strip()}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


================================================================================
ZERO-SHOT INDIAN FOOD REVIEW SENTIMENT ANALYSIS
================================================================================

Food Review: The Punjabi Amritsari kulcha was crispy, buttery, and absolutely delicious.
Sentiment: POSITIVE

Food Review: The Tamil Nadu masala dosa was cold, soggy, and disappointing.
Sentiment: NEGATIVE

Food Review: The Bengali mishti doi was served after the main meal.
Sentiment: NEUTRAL

Food Review: The Gujarati dhokla was fine, but there was nothing special about it.
Sentiment: NEUTRAL

Food Review: The Hyderabadi dum biryani was aromatic, flavourful, and perfectly spiced.
Sentiment: POSITIVE



"""
