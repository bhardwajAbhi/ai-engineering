"""
Title: ReAct Prompting (Google Generative AI)

ReAct lets the model alternate between Thought (reasoning), Action (using
a tool) and Observation (the result of the tool) until it can give a final
answer. The tools here are simulated, so no real search is made.

uv run python src/19_react_google.py
"""

import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Simulated tools
def search(query: str) -> str:
    """Simulated web search."""
    fake_results = {
        "dpdp penalty": "Under India's DPDP Act, 2023, the penalty for failing to take reasonable security safeguards against a personal data breach can go up to ₹250 crore.",
        "protection penalty": "Under India's DPDP Act, 2023, the penalty for failing to take reasonable security safeguards against a personal data breach can go up to ₹250 crore.",
        "dpdp child": "Under India's DPDP Act, 2023, a child is a person who has not completed 18 years of age.",
        "gdpr breach": "The GDPR requires a breach to be reported to the supervisory authority within 72 hours of becoming aware of it, where feasible.",
    }
    query = query.lower()
    for key, value in fake_results.items():
        # A result matches when every word of the key is in the query
        if all(word in query for word in key.split()):
            return value
    return f"No specific results found for: {query}"


def calculate(expression: str) -> str:
    """Simple calculator. Only numbers and + - * / ( ) are allowed."""
    if not re.fullmatch(r"[0-9+\-*/(). ]+", expression):
        return f"Could not calculate: {expression}"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return f"Could not calculate: {expression}"


tools = {"search": search, "calculate": calculate}


def run_react(question: str, react_chain, continue_chain, max_steps: int = 5) -> str:
    """Run the ReAct loop."""

    # Initial prompt
    response = react_chain.invoke({"question": question})
    full_trace = response

    for step in range(max_steps):
        # Check if we have a final answer
        if "Answer:" in response and "Action:" not in response.split("Answer:")[-1]:
            break

        # Find the action
        action_match = re.search(r"Action:\s*(\w+)\((.+)\)", response)
        if not action_match:
            break

        tool_name = action_match.group(1)
        tool_input = action_match.group(2).strip("\"'")

        # Execute the tool
        # The model only writes the tool name as text. Our code finds the
        # matching function in the tools dictionary and runs it.
        if tool_name in tools:
            observation = tools[tool_name](tool_input)
        else:
            observation = f"Unknown tool: {tool_name}"

        # The same tool calling written step by step, without the dictionary.
        # It works the same way, but every new tool needs one more elif.
        #
        # if tool_name == "search":
        #     observation = search(tool_input)
        # elif tool_name == "calculate":
        #     observation = calculate(tool_input)
        # elif tool_name == "new_tool":  # a new tool needs one more elif like this
        #     observation = new_tool(tool_input)
        # else:
        #     observation = f"Unknown tool: {tool_name}"

        print(f"Step {step + 1}")
        print(f"Action: {tool_name}({tool_input})")
        print(f"Observation: {observation}")

        # Continue with the observation
        response = continue_chain.invoke(
            {"question": question, "previous": full_trace, "observation": observation}
        )
        full_trace += f"\nObservation: {observation}\n{response}"

    return full_trace


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    react_prompt = ChatPromptTemplate.from_template("""
        Answer the question using the ReAct pattern: Thought, Action, Observation, repeat until done.

        Available tools:
        - search(query): Search for information
        - calculate(expression): Evaluate math expressions

        Format your response as:
        Thought: [your reasoning]
        Action: [tool_name(input)]
        Observation: [will be filled by tool result]
        ... (repeat as needed)
        Thought: [final reasoning]
        Answer: [your final answer]

        After writing an Action, stop and wait. Do not write the Observation yourself.

        Question: {question}

        Begin:
    """)

    continue_prompt = ChatPromptTemplate.from_template("""
        Question: {question}

        {previous}
        Observation: {observation}

        Continue your reasoning. If you have enough information, provide the final Answer.
    """)

    # Create the two chains
    react_chain = react_prompt | llm | parser
    continue_chain = continue_prompt | llm | parser

    # Test questions: one needs a search, the other needs a search and a calculation
    questions = [
        "What is the maximum penalty under India's DPDP Act, 2023, for failing to take reasonable security safeguards against a data breach?",
        "A company is fined the maximum DPDP Act penalty for failing to protect personal data in each of 3 separate breaches. How many crore rupees is that in total?",
    ]

    for i, question in enumerate(questions, 1):
        print("=" * 60)
        print(f"QUESTION {i}: {question}")
        print("-" * 60)
        trace = run_react(question, react_chain, continue_chain)
        print("\nFULL TRACE:")
        print(trace)
        print()


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
QUESTION 1: What is the maximum penalty under India's DPDP Act, 2023, for failing to take reasonable security safeguards against a data breach?
------------------------------------------------------------
Step 1
Action: search(maximum penalty India DPDP Act 2023 failure to take reasonable security safeguards)
Observation: Under India's DPDP Act, 2023, the penalty for failing to take reasonable security safeguards against a personal data breach can go up to ₹250 crore.

FULL TRACE:
Thought: I need to find the specific penalty amount mentioned in India's Digital Personal Data Protection (DPDP) Act, 2023, for the failure to implement reasonable security safeguards to prevent data breaches.

Action: search("maximum penalty India DPDP Act 2023 failure to take reasonable security safeguards")
Observation: Under India's DPDP Act, 2023, the penalty for failing to take reasonable security safeguards against a personal data breach can go up to ₹250 crore.
The maximum penalty under India's Digital Personal Data Protection (DPDP) Act, 2023, for failing to take reasonable security safeguards to prevent a personal data breach is **₹250 crore**.

============================================================
QUESTION 2: A company is fined the maximum DPDP Act penalty for failing to protect personal data in each of 3 separate breaches. How many crore rupees is that in total?
------------------------------------------------------------
Step 1
Action: search(maximum penalty under DPDP Act India)
Observation: Under India's DPDP Act, 2023, the penalty for failing to take reasonable security safeguards against a personal data breach can go up to ₹250 crore.

FULL TRACE:
Thought: I need to determine the maximum penalty amount under the Digital Personal Data Protection (DPDP) Act of India for a single breach and then multiply that by 3.

Action: search("maximum penalty under DPDP Act India")
Observation: Under India's DPDP Act, 2023, the penalty for failing to take reasonable security safeguards against a personal data breach can go up to ₹250 crore.
The maximum penalty for failing to take reasonable security safeguards to prevent a personal data breach under the Digital Personal Data Protection (DPDP) Act, 2023, is ₹250 crore per instance.

Since the company was fined this maximum amount for each of the 3 separate breaches, the calculation is as follows:
₹250 crore × 3 = ₹750 crore.

**Answer:** The total penalty is 750 crore rupees.



"""