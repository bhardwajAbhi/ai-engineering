"""
Title: ReAct Prompting (OpenRouter)

ReAct lets the model alternate between Thought (reasoning), Action (using
a tool) and Observation (the result of the tool) until it can give a final
answer. The tools here are simulated, so no real search is made.

uv run python src/19_react_openrouter.py
"""

import re

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Simulated tools
def search(query: str) -> str:
    """Simulated web search."""
    fake_results = {
        "sha256 hash": "SHA-256 produces a hash that is 256 bits long.",
        "aes block": "AES works on blocks of 128 bits (16 bytes).",
        "https port": "HTTPS uses port 443 by default.",
    }
    # Ignore hyphens so "SHA-256" and "SHA256" both match
    query = query.lower().replace("-", "")
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
    # Some models write the Observation themselves after an Action. Keep only the
    # text before it, so our code runs the real tool instead of trusting an invented result.
    if "Action:" in response:
        response = response.split("Observation:")[0]
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
        # Same as above: drop any Observation the model made up after an Action
        if "Action:" in response:
            response = response.split("Observation:")[0]
        full_trace += f"\nObservation: {observation}\n{response}"

    return full_trace


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="meta-llama/llama-3.1-8b-instruct",
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

        Continue your reasoning. If the Observation already gives you what you need,
        write the final Answer.
    """)

    # Create the two chains
    react_chain = react_prompt | llm | parser
    continue_chain = continue_prompt | llm | parser

    # Test questions: one needs a search, the other needs a search and a calculation
    questions = [
        "How many bits long is a SHA-256 hash?",
        "A SHA-256 hash is written as hexadecimal text, where each character stands for 4 bits. How many characters long is that text?",
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
QUESTION 1: How many bits long is a SHA-256 hash?
------------------------------------------------------------
Step 1
Action: search(SHA-256 hash length in bits)
Observation: SHA-256 produces a hash that is 256 bits long.

FULL TRACE:
Thought: I need to confirm the length of a SHA-256 hash. Let me search for this information.
Action: search("SHA-256 hash length in bits")
Observation: SHA-256 produces a hash that is 256 bits long.
SHA-256 produces a hash that is **256 bits** long.

============================================================
QUESTION 2: A SHA-256 hash is written as hexadecimal text, where each character stands for 4 bits. How many characters long is that text?
------------------------------------------------------------
Step 1
Action: calculate(256 / 4)
Observation: 64.0

FULL TRACE:
Thought: SHA-256 produces a 256-bit hash. Each hexadecimal character represents 4 bits. Therefore, the number of characters is 256 divided by 4.
Action: calculate(256 / 4)

Observation: 64.0
A SHA-256 hash is 256 bits long. Since each hexadecimal character represents 4 bits, the number of characters is 256 ÷ 4 = **64**.

**Answer: 64**



"""