"""
Title: Structured Output with Pydantic (Google Generative AI)
uv run python src/06_pydantic_structured_output_google.py
"""
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


class DataCollectionPractice(BaseModel):
    """Details about one category of personal data mentioned in the policy."""

    data_category: str = Field(description="General category of personal data")

    data_elements: list[str] = Field(description="Specific personal data elements collected")

    collection_method: Literal["USER_PROVIDED", "AUTOMATICALLY_COLLECTED", "RECEIVED_FROM_THIRD_PARTY", "NOT_SPECIFIED"]

    purposes: list[str] = Field(description="Purpose for which the data is collected or used")


class PrivacyPolicyAnalysis(BaseModel):
    """Structured analysis of a mobile application's privacy policy."""

    app_name: str = Field(description="Name of the mobile application")

    data_practices: list[DataCollectionPractice] = Field(description="Personal data categories and their collection practices")

    third_party_sharing: list[str] = Field(description="Third parties or recipient categories receiving the data")

    retention_periods: list[str] = Field(description="Data retention periods stated in the policy")

    user_controls: list[str] = Field(description="Privacy controls or choices available to users")

    security_measures: list[str] = Field(description="Security measures claimed in the policy")

    privacy_concerns: list[str] = Field(description="Important ambiguities, omissions, or privacy concerns")

    overall_transparency: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="How clearly the policy explains its data practices")

    summary: str = Field(description="A concise summary of the application's privacy practices")

    


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    print("="*80)
    print("MOBILE APP PRIVACY POLICY ANALYSIS")
    print("="*80)

    # The parser generates formatting instructions from the Pydantic models
    # and converts the JSON response into a Python dictionary.
    parser = JsonOutputParser(pydantic_object=PrivacyPolicyAnalysis)

    #Prompt for analysing the privacy policy
    privacy_policy_prompt = ChatPromptTemplate.from_template("""
        You are an experienced mobile application privacy analyst.

        Read the supplied privacy policy and extract its data practices.

        Important instructions:

        1. Report only information stated or reasonably evident in the policy.
        2. Do not assume that the application collects data that is not mentioned.
        3. Distinguish between user-provided data and automatically collected data.
        4. Identify unclear statements, missing details, and broad data-use purposes.
        5. This is a preliminary policy analysis, not a final legal compliance decision.

        {format_instructions}

        Privacy policy:

        {privacy_policy}
    """)

    # Create the chain
    chain = privacy_policy_prompt | llm | parser

    # Sample privacy policy
    privacy_policy = """
        QuickRide Privacy Policy

        QuickRide is a mobile application that allows users to book local taxi rides.

        Information provided by users:
        When an account is created, we collect the user's full name, mobile number,
        email address, and profile photograph. Users may optionally add an emergency
        contact containing the contact's name and mobile number.

        Location and trip information:
        We collect precise location while the application is being used to identify
        nearby drivers, determine pickup and destination points, calculate fares, and
        provide navigation. We maintain a record of completed rides, including pickup
        location, destination, fare, driver details, date, and time.

        Payment information:
        Payments are processed by our payment service provider. QuickRide stores the
        payment transaction identifier and payment status but does not store complete
        credit or debit card numbers.

        Device and usage information:
        We automatically collect the device model, operating-system version, IP
        address, advertising identifier, application version, crash reports, and
        information about how users interact with the application.

        How we use information:
        We use personal data to create and manage accounts, provide rides, process
        payments, prevent fraud, provide customer support, improve our services, and
        send promotional offers. Users may disable promotional notifications through
        the application settings.

        Sharing of information:
        Pickup location, destination, first name, and masked phone number are shared
        with the assigned driver. Payment information is shared with our payment
        processor. Device and usage information may be shared with analytics and
        advertising partners. We may also disclose information when required by law.

        Retention:
        Account information is retained while the account remains active. Trip and
        transaction records may be retained for seven years for legal and accounting
        purposes. Crash reports are normally retained for 12 months. The policy does
        not specify how long advertising identifiers or precise location records are
        retained.

        User choices:
        Users may view and correct their profile information, disable promotional
        notifications, revoke location permission through device settings, or request
        account deletion by contacting privacy@quickride.example. Some records may
        continue to be retained when required by law.

        Security:
        QuickRide states that it uses encryption during transmission, access controls,
        and periodic security reviews. However, no information is provided about
        encryption of stored personal data.

        Policy updates:
        This policy may be updated periodically. Users will be informed about material
        changes through the application or by email.
        """


    # Invoke the chain
    result = chain.invoke({"privacy_policy":privacy_policy, "format_instructions": parser.get_format_instructions()})

    # Display raw results
    print(f"\nResult object type: {type(result)}")
    print(f"Result content: {result}")
    print("-" * 80)

    # Display formatted results
    print(f"\nApplication: {result['app_name']}")
    print(f"Overall Transparency: {result['overall_transparency']}")

    print("\nPERSONAL DATA COLLECTION")
    print("-" * 80)

    for practice in result["data_practices"]:
        print(f"\nData Category: {practice['data_category']}")
        print(f"Collection Method: {practice['collection_method']}")

        print("Data Elements:")
        for element in practice["data_elements"]:
            print(f"  - {element}")

        print("Purposes:")
        for purpose in practice["purposes"]:
            print(f"  - {purpose}")

    print("\nTHIRD-PARTY SHARING")
    print("-" * 80)
    for recipient in result["third_party_sharing"]:
        print(f"  - {recipient}")

    print("\nRETENTION PERIODS")
    print("-" * 80)
    for retention in result["retention_periods"]:
        print(f"  - {retention}")

    print("\nUSER CONTROLS")
    print("-" * 80)
    for control in result["user_controls"]:
        print(f"  - {control}")

    print("\nSECURITY MEASURES")
    print("-" * 80)
    for measure in result["security_measures"]:
        print(f"  - {measure}")

    print("\nPRIVACY CONCERNS")
    print("-" * 80)
    for concern in result["privacy_concerns"]:
        print(f"  - {concern}")

    print("\nSUMMARY")
    print("-" * 80)
    print(result["summary"])



if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------

================================================================================
MOBILE APP PRIVACY POLICY ANALYSIS
================================================================================
Result object type: <class 'dict'>
Result content: {'app_name': 'QuickRide', 'data_practices': [{'data_category': 'Account Information', 'data_elements': ['full name', 'mobile number', 'email address', 'profile photograph'], 'collection_method': 'USER_PROVIDED', 'purposes': ['create and manage accounts', 'provide customer support']}, {'data_category': 'Emergency Contact Information', 'data_elements': ['contact name', 'contact mobile number'], 'collection_method': 'USER_PROVIDED', 'purposes': ['emergency contact']}, {'data_category': 'Location and Trip Information', 'data_elements': ['precise location', 'pickup location', 'destination', 'fare', 'driver details', 'date', 'time'], 'collection_method': 'AUTOMATICALLY_COLLECTED', 'purposes': ['identify nearby drivers', 'determine pickup and destination points', 'calculate fares', 'provide navigation', 'provide rides']}, {'data_category': 'Payment Information', 'data_elements': ['payment transaction identifier', 'payment status'], 'collection_method': 'AUTOMATICALLY_COLLECTED', 'purposes': ['process payments']}, {'data_category': 'Device and Usage Information', 'data_elements': ['device model', 'operating-system version', 'IP address', 'advertising identifier', 'application version', 'crash reports', 'interaction information'], 'collection_method': 'AUTOMATICALLY_COLLECTED', 'purposes': ['prevent fraud', 'improve services', 'send promotional offers']}], 'third_party_sharing': ['assigned drivers', 'payment service provider', 'analytics partners', 'advertising partners', 'legal authorities'], 'retention_periods': ['Account information: while account is active', 'Trip and transaction records: 7 years', 'Crash reports: 12 months'], 'user_controls': ['view and correct profile information', 'disable promotional notifications', 'revoke location permission', 'request account deletion'], 'security_measures': ['encryption during transmission', 'access controls', 'periodic security reviews'], 'privacy_concerns': ['Retention period for advertising identifiers is not specified', 'Retention period for precise location records is not specified', 'Lack of information regarding encryption of stored personal data', "Broad purpose for 'improving services' and 'preventing fraud' without specific details"], 'overall_transparency': 'MEDIUM', 'summary': 'QuickRide collects user-provided account details and automatically tracks location, trip history, and device usage to facilitate taxi services, process payments, and serve advertisements. While it provides clear retention periods for some data, it fails to specify retention for location and advertising data and lacks detail on data-at-rest security.'}
--------------------------------------------------------------------------------

Application: QuickRide
Overall Transparency: MEDIUM

PERSONAL DATA COLLECTION
--------------------------------------------------------------------------------

Data Category: Account Information
Collection Method: USER_PROVIDED
Data Elements:
  - full name
  - mobile number
  - email address
  - profile photograph
Purposes:
  - create and manage accounts
  - provide customer support

Data Category: Emergency Contact Information
Collection Method: USER_PROVIDED
Data Elements:
  - contact name
  - contact mobile number
Purposes:
  - emergency contact

Data Category: Location and Trip Information
Collection Method: AUTOMATICALLY_COLLECTED
Data Elements:
  - precise location
  - pickup location
  - destination
  - fare
  - driver details
  - date
  - time
Purposes:
  - identify nearby drivers
  - determine pickup and destination points
  - calculate fares
  - provide navigation
  - provide rides

Data Category: Payment Information
Collection Method: AUTOMATICALLY_COLLECTED
Data Elements:
  - payment transaction identifier
  - payment status
Purposes:
  - process payments

Data Category: Device and Usage Information
Collection Method: AUTOMATICALLY_COLLECTED
Data Elements:
  - device model
  - operating-system version
  - IP address
  - advertising identifier
  - application version
  - crash reports
  - interaction information
Purposes:
  - prevent fraud
  - improve services
  - send promotional offers

THIRD-PARTY SHARING
--------------------------------------------------------------------------------
  - assigned drivers
  - payment service provider
  - analytics partners
  - advertising partners
  - legal authorities

RETENTION PERIODS
--------------------------------------------------------------------------------
  - Account information: while account is active
  - Trip and transaction records: 7 years
  - Crash reports: 12 months

USER CONTROLS
--------------------------------------------------------------------------------
  - view and correct profile information
  - disable promotional notifications
  - revoke location permission
  - request account deletion

SECURITY MEASURES
--------------------------------------------------------------------------------
  - encryption during transmission
  - access controls
  - periodic security reviews

PRIVACY CONCERNS
--------------------------------------------------------------------------------
  - Retention period for advertising identifiers is not specified
  - Retention period for precise location records is not specified
  - Lack of information regarding encryption of stored personal data
  - Broad purpose for 'improving services' and 'preventing fraud' without specific details

SUMMARY
--------------------------------------------------------------------------------
QuickRide collects user-provided account details and automatically tracks location, trip history, and device usage to facilitate taxi services, process payments, and serve advertisements. While it provides clear retention periods for some data, it fails to specify retention for location and advertising data and lacks detail on data-at-rest security.


"""