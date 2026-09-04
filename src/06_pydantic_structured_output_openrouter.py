"""
Title: Structured Output with Pydantic (OpenRouter)
uv run python src/06_pydantic_structured_output_openrouter.py
"""

from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openrouter import ChatOpenRouter


class ProductReview(BaseModel):
    """Structured analysis of a product review"""

    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(description="Overall sentiment expressed in the product review")

    confidence: float = Field(description="Confidence score for the sentiment classification", ge=0.0, le=1.0)

    key_points: list[str] = Field(description="Two or three main points identified in the review", min_length=2, max_length=3)

    suggested_action: str = Field(description="A concise recommended for the business")


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    print("=" * 60)
    print("STRUCTURED PRODUCT REVIEW ANALYSIS")
    print("=" * 60)

    # Create the parser using Pydantic model
    parser = JsonOutputParser(pydantic_object=ProductReview) 

    # Create the prompt
    review_analysis_prompt = ChatPromptTemplate.from_template("""
        You are an experienced customer-feedback analyst. 
        Analyze the product review and extract the required information.

        {format_instructions}

        Review: {review}
    """)

    # Create the chain
    chain = review_analysis_prompt | llm | parser


    # Some sample reviews
    reviews = [
        """
        The battery life is incredible and easily lasts three days.
        However, the camera struggles in low light. Overall, I am
        very happy with my purchase.
        """,

        """
        Worst phone I have ever owned. It crashes constantly,
        overheats, and the screen cracked within a week.
        Avoid it at all costs!
        """,

        """
        It is an okay phone. It does what it needs to do.
        Nothing special, but I have no major complaints either.
        """,
    ]

    # Analyse each product review
    for review in reviews:
        cleaned_review = " ".join(review.split())

        print(f"\n{'=' * 60}")
        print(f"Review: {cleaned_review}")
        print("-" * 60)

        result = chain.invoke(
            {
                "review": cleaned_review,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")

        print("Key Points:")
        for point in result["key_points"]:
            print(f"  - {point}")

        print(f"Suggested Action: {result['suggested_action']}")
        print(f"Result Type: {type(result)}")
        print(f"Raw Result: {result}")





if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

============================================================
STRUCTURED PRODUCT REVIEW ANALYSIS
============================================================

============================================================
Review: The battery life is incredible and easily lasts three days. However, the camera struggles in low light. Overall, I am very happy with my purchase.
------------------------------------------------------------
Sentiment: POSITIVE
Confidence: 0.85
Key Points:
  - Battery life is incredible and lasts three days
  - Camera struggles in low light
  - Overall happy with purchase
Suggested Action: Improve camera performance in low light conditions
Result Type: <class 'dict'>
Raw Result: {'sentiment': 'POSITIVE', 'confidence': 0.85, 'key_points': ['Battery life is incredible and lasts three days', 'Camera struggles in low light', 'Overall happy with purchase'], 'suggested_action': 'Improve camera performance in low light conditions'}

============================================================
Review: Worst phone I have ever owned. It crashes constantly, overheats, and the screen cracked within a week. Avoid it at all costs!
------------------------------------------------------------
Sentiment: NEGATIVE
Confidence: 0.98
Key Points:
  - crashes constantly
  - overheats
  - screen cracked within a week
Suggested Action: Improve product quality and reliability
Result Type: <class 'dict'>
Raw Result: {'sentiment': 'NEGATIVE', 'confidence': 0.98, 'key_points': ['crashes constantly', 'overheats', 'screen cracked within a week'], 'suggested_action': 'Improve product quality and reliability'}

============================================================
Review: It is an okay phone. It does what it needs to do. Nothing special, but I have no major complaints either.
------------------------------------------------------------
Sentiment: NEUTRAL
Confidence: 0.8
Key Points:
  - okay phone
  - does what it needs to do
  - no major complaints
Suggested Action: Maintain current pricing and features
Result Type: <class 'dict'>
Raw Result: {'sentiment': 'NEUTRAL', 'confidence': 0.8, 'key_points': ['okay phone', 'does what it needs to do', 'no major complaints'], 'suggested_action': 'Maintain current pricing and features'}

"""