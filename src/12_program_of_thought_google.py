"""
Title: Program of Thought Prompting (Google Generative AI)
Program-of-Thought prompting asks the model to solve a problem by
writing Python code instead of reasoning in plain text, so the actual
arithmetic is done by a real interpreter, not guessed by the model.

This version has the model print() the final answer, then captures
that printed output by temporarily redirecting stdout.

uv run python src/12_program_of_thought_google.py
"""

import io
import re
import sys

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_and_run_code(response: str) -> str:
    """Extract Python code from the response and execute it, capturing
    whatever the code prints to stdout."""

    # Find the fenced code block in the model's response
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        return f"No code found. Raw response:\n{response}"

    code = code_match.group(1)

    # Execute the extracted code and capture its printed output
    try:
        # Save the real stdout, then point it to an in-memory buffer
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        exec(code, {"__builtins__": __builtins__})

        # Read what the code printed, then restore the real stdout
        output = buffer.getvalue()
        sys.stdout = old_stdout
        return f"CODE:\n{code}\n\nOUTPUT:\n{output}"
    except Exception as e:
        return f"Error executing code: {e}\n\nCODE:\n{code}"


# ---------------------------------------------------------------------
# Alternative style: contextlib.redirect_stdout
# Same idea as above, but the "with" block restores stdout for us
# automatically, even if the executed code raises an error.
#
# from contextlib import redirect_stdout
#
# def extract_and_run_code(response: str) -> str:
#     code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
#     if not code_match:
#         return f"No code found. Raw response:\n{response}"
#
#     code = code_match.group(1)
#
#     try:
#         buffer = io.StringIO()
#         with redirect_stdout(buffer):
#             exec(code, {"__builtins__": __builtins__})
#
#         output = buffer.getvalue()
#         return f"CODE:\n{code}\n\nOUTPUT:\n{output}"
#     except Exception as e:
#         return f"Error executing code: {e}\n\nCODE:\n{code}"
# ---------------------------------------------------------------------


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Program-of-Thought prompt
    pot_prompt = ChatPromptTemplate.from_template("""
        You are an experienced cybersecurity analyst.

        Solve this problem by writing Python code.
        Think through the problem, then write code that computes the answer.
        Put your code in a ```python``` block.

        PROBLEM: {problem}

        First, explain your approach briefly.
        Then write the Python code.
        The code should print the final answer.
    """)

    # Create the chain
    chain = pot_prompt | llm | parser

    # Sample security problems
    problems = [
        """
        A SOC ingests 12,400 log events per hour. During a DDoS attack,
        log volume spikes to 4.5 times the normal rate for 3 hours before
        returning to normal. How many total log events were ingested
        during those 3 hours of the attack?
        """,
        """
        A company's incident response retainer includes 40 hours of
        support per quarter at no extra cost. A ransomware incident took
        65 hours to resolve. Extra hours beyond the retainer are billed
        at $350/hour. What was the total extra cost billed for this
        incident?
        """,
    ]

    # Invoke the chain for each problem, one at a time
    for i, problem in enumerate(problems, 1):
        print(f"\n{'=' * 60}")
        print(f"PROBLEM {i}")
        print("=" * 60)
        print(f"\n{problem.strip()}\n")
        print("-" * 60)

        # Get the raw response from the model
        response = chain.invoke({"problem": problem})

        # Show the original model response before any code is extracted
        print("ORIGINAL MODEL RESPONSE:")
        print(response)
        print("\n" + "-" * 60)

        # Extract the code from the response and run it
        result = extract_and_run_code(response)
        print("EXTRACTED CODE AND EXECUTION RESULT:")
        print(result)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
PROBLEM 1
============================================================

A SOC ingests 12,400 log events per hour. During a DDoS attack,
        log volume spikes to 4.5 times the normal rate for 3 hours before
        returning to normal. How many total log events were ingested
        during those 3 hours of the attack?

------------------------------------------------------------
ORIGINAL MODEL RESPONSE:
To solve this problem, we need to calculate the total number of log events ingested during the 3-hour DDoS attack window.

### Approach:
1.  **Identify the normal rate:** 12,400 events per hour.
2.  **Calculate the attack rate:** Multiply the normal rate by the spike factor (4.5).
3.  **Calculate the total volume:** Multiply the attack rate by the duration of the attack (3 hours).

### Python Code:

```python
# Define the variables
normal_rate_per_hour = 12400
spike_factor = 4.5
attack_duration_hours = 3

# Calculate the rate during the attack
attack_rate_per_hour = normal_rate_per_hour * spike_factor

# Calculate the total events during the 3-hour window
total_events_during_attack = attack_rate_per_hour * attack_duration_hours

# Print the result
print(f"Total log events ingested during the 3-hour attack: {int(total_events_during_attack)}")
```

------------------------------------------------------------
EXTRACTED CODE AND EXECUTION RESULT:
CODE:
# Define the variables
normal_rate_per_hour = 12400
spike_factor = 4.5
attack_duration_hours = 3

# Calculate the rate during the attack
attack_rate_per_hour = normal_rate_per_hour * spike_factor

# Calculate the total events during the 3-hour window
total_events_during_attack = attack_rate_per_hour * attack_duration_hours

# Print the result
print(f"Total log events ingested during the 3-hour attack: {int(total_events_during_attack)}")


OUTPUT:
Total log events ingested during the 3-hour attack: 167400


============================================================
PROBLEM 2
============================================================

A company's incident response retainer includes 40 hours of
        support per quarter at no extra cost. A ransomware incident took
        65 hours to resolve. Extra hours beyond the retainer are billed
        at $350/hour. What was the total extra cost billed for this
        incident?

------------------------------------------------------------
ORIGINAL MODEL RESPONSE:
To calculate the total extra cost billed for the incident, we follow these steps:

1.  **Identify the total hours spent:** 65 hours.
2.  **Identify the retainer limit:** 40 hours.
3.  **Calculate the billable hours:** Subtract the retainer hours from the total hours spent (65 - 40).
4.  **Calculate the cost:** Multiply the billable hours by the hourly rate ($350).

Here is the Python code to perform this calculation:

```python
# Incident Response Cost Calculation

# Define the variables
total_hours_spent = 65
retainer_hours = 40
hourly_rate = 350

# Calculate billable hours
billable_hours = total_hours_spent - retainer_hours

# Calculate total extra cost
total_extra_cost = billable_hours * hourly_rate

# Print the result
print(f"Total extra cost billed: ${total_extra_cost}")
```

------------------------------------------------------------
EXTRACTED CODE AND EXECUTION RESULT:
CODE:
# Incident Response Cost Calculation

# Define the variables
total_hours_spent = 65
retainer_hours = 40
hourly_rate = 350

# Calculate billable hours
billable_hours = total_hours_spent - retainer_hours

# Calculate total extra cost
total_extra_cost = billable_hours * hourly_rate

# Print the result
print(f"Total extra cost billed: ${total_extra_cost}")


OUTPUT:
Total extra cost billed: $8750


"""