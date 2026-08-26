"""
Title: Few Shot Prompting (OpenRouter)
uv run python src/05_few_shot_openrouter.py
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
    
    print("="*60)
    print("FEW-SHOT MOBILE APP PRIVACY ANALYSIS")
    print("="*60)

    # Few-shot prompt with mobile application privacy examples
    privacy_prompt = ChatPromptTemplate.from_messages(
        [
            # "system", "human", and "ai" are predefined message roles.
            (
                "system",
                """You are an experienced mobile application security analyst
                    who performs privacy-risk analysis of mobile applications.

                    Classify each observation using exactly one label:

                    CONSENT_CONCERN
                    PURPOSE_CONCERN
                    SECURITY_CONCERN
                    EXPECTED_USE

                    This is a preliminary privacy assessment, not a final legal decision.
                    Respond with only the classification label.""",
            ),

            # Few-shot Example 1: Consent concern
            ("human","The user denied contact permission, but the app still reads the contact list."), # "human" provides the example input.
            ("ai", "CONSENT_CONCERN"), # "ai" provides the expected response for the example.

            # Few-shot Example 2: Purpose concern
            ("human", "A delivery app uses location data collected for delivery to create advertising profiles without informing the user."),
            ("ai", "PURPOSE_CONCERN"),

            # Few-shot Example 3: Security concern
            ("human", "The app sends users' phone numbers to its server using an unencrypted connection."),
            ("ai", "SECURITY_CONCERN"),

            # Few-shot Example 4: Expected use
            ("human", "A camera app accesses the camera only after the user grants permission and taps Take Photo."),
            ("ai", "EXPECTED_USE"),

            # Few-shot Example 5: Consent concern
            ("human", "The app starts recording audio before asking the user for microphone permission."),
            ("ai", "CONSENT_CONCERN"),

            # Actual query that will be supplied during invocation
            ("human", "{query}"),
        ]
    )

    # The Chain
    chain = privacy_prompt | llm | parser
    
    # Sample mobile application observations
    sample_observations = [
        "A shopping app uploads the user's contacts after contact permission was denied.",
        "A weather app uses location data for marketing without informing the user.",
        "A health app stores the user's name and medical report in an unprotected file.",
        "A food delivery app uses the address entered by the user to deliver the current order.",
        "A flashlight app collects a device identifier for targeted advertising.",
    ]

    # Invoke the chain for each observation
    for observation in sample_observations:
        result = chain.invoke({"query": observation})

        print(f"\nObservation: {observation}")
        print(f"Classification: {result.strip()}")



if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
FEW-SHOT MOBILE APP PRIVACY ANALYSIS
============================================================

Observation: A shopping app uploads the user's contacts after contact permission was denied.
Classification: CONSENT_CONCERN

Observation: A weather app uses location data for marketing without informing the user.
Classification: PURPOSE_CONCERN

Observation: A health app stores the user's name and medical report in an unprotected file.
Classification: SECURITY_CONCERN

Observation: A food delivery app uses the address entered by the user to deliver the current order.
Classification: EXPECTED_USE

Observation: A flashlight app collects a device identifier for targeted advertising.
Classification: PURPOSE_CONCERN



"""