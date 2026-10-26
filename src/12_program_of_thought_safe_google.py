"""
Title: Program of Thought Prompting, Safe Execution (Google Generative AI)
Program-of-Thought prompting asks the model to solve a problem by
writing Python code instead of reasoning in plain text, so the actual
arithmetic is done by a real interpreter, not guessed by the model.

The basic version runs the model's code with exec() inside our own
program. That code is untrusted: a hidden instruction in the input can
make the model write code that reads files, runs commands or never ends.

This version adds a basic safety layer before anything runs:
  1. Check the code with the ast module and reject risky imports,
     names and attributes.
  2. Run it in a separate Python process, not inside our program.
  3. Limit that process: a timeout, an empty environment (no API keys),
     an empty temporary folder and Python's isolated mode (-I).

The model prints the final answer, and we read it from the captured
output of the separate process. Our own stdout is never redirected.

This is a basic fix for learning, not a full sandbox. Real systems add
containers, OS-level limits or dedicated sandboxes on top.

uv run python src/12_program_of_thought_safe_google.py
"""

import ast
import re
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Modules that a math answer may need. Everything else is rejected.
ALLOWED_IMPORTS = {"math", "statistics", "datetime"}

# Built-in names that a math answer never needs, but an attacker would.
BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint",
}

# Seconds before we stop the code (stops endless loops)
TIMEOUT_SECONDS = 5


def check_code(code: str) -> None:
    """Read the code as a syntax tree (without running it) and raise
    ValueError if it uses anything outside our small safe set."""

    tree = ast.parse(code)

    for node in ast.walk(tree):
        # 1. Only allow imports from the allowlist
        if isinstance(node, ast.Import):
            names = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            names = set()

        blocked = names - ALLOWED_IMPORTS
        if blocked:
            raise ValueError(f"Import not allowed: {', '.join(sorted(blocked))}")

        # 2. Block risky built-in functions such as open() or eval()
        if isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            raise ValueError(f"Name not allowed: {node.id}")

        # 3. Block attributes that start with "_" such as __class__,
        #    a common way to climb back to dangerous objects
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"Attribute not allowed: {node.attr}")


def run_code_safely(code: str) -> str:
    """Check the code, then run it in a separate, limited Python process
    and return what it printed."""

    try:
        check_code(code)
    except (SyntaxError, ValueError) as e:
        return f"Code rejected: {e}"

    # An empty temporary folder as the working directory
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],  # -I: isolated mode
                capture_output=True,                  # collect stdout and stderr
                text=True,
                timeout=TIMEOUT_SECONDS,              # stop endless loops
                cwd=workdir,                          # no access to our files by path
                env={},                               # no environment variables, no API keys
            )
        except subprocess.TimeoutExpired:
            return f"Code stopped after {TIMEOUT_SECONDS} seconds."

    if result.returncode != 0:
        return f"Code failed:\n{result.stderr.strip()}"

    return result.stdout


def extract_and_run_code(response: str) -> str:
    """Extract Python code from the response and run it safely."""

    # Find the fenced code block in the model's response
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        return f"No code found. Raw response:\n{response}"

    code = code_match.group(1)
    output = run_code_safely(code)
    return f"CODE:\n{code}\n\nOUTPUT:\n{output}"


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
        Use only plain Python and, if needed, the math module.

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

        # Extract the code from the response and run it safely
        result = extract_and_run_code(response)
        print("EXTRACTED CODE AND SAFE EXECUTION RESULT:")
        print(result)

    # Show what the safety layer does with harmful code
    # (no model call here, these are fixed test inputs)
    print(f"\n{'=' * 60}")
    print("SAFETY CHECK DEMO")
    print("=" * 60)

    harmful_samples = {
        "Reads a secret file": 'print(open(".env").read())',
        "Runs a shell command": 'import os\nos.system("whoami")',
        "Climbs to hidden objects": "print(().__class__.__bases__[0].__subclasses__())",
        "Never ends": "while True:\n    pass",
    }

    for label, code in harmful_samples.items():
        print(f"\n{label}:")
        print(run_code_safely(code))


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
1.  **Identify the normal rate:** The normal ingestion rate is 12,400 events per hour.
2.  **Calculate the attack rate:** The attack rate is 4.5 times the normal rate.
3.  **Calculate the total volume:** Multiply the attack rate by the duration of the attack (3 hours).

### Python Code:

```python
# Define the normal ingestion rate per hour
normal_rate_per_hour = 12400

# Define the multiplier for the DDoS attack
attack_multiplier = 4.5

# Define the duration of the attack in hours
attack_duration_hours = 3

# Calculate the attack rate per hour
attack_rate_per_hour = normal_rate_per_hour * attack_multiplier

# Calculate the total events during the 3-hour attack
total_events_during_attack = attack_rate_per_hour * attack_duration_hours

# Print the result
print(f"Total log events ingested during the 3 hours of the attack: {int(total_events_during_attack)}")
```

------------------------------------------------------------
EXTRACTED CODE AND SAFE EXECUTION RESULT:
CODE:
# Define the normal ingestion rate per hour
normal_rate_per_hour = 12400

# Define the multiplier for the DDoS attack
attack_multiplier = 4.5

# Define the duration of the attack in hours
attack_duration_hours = 3

# Calculate the attack rate per hour
attack_rate_per_hour = normal_rate_per_hour * attack_multiplier

# Calculate the total events during the 3-hour attack
total_events_during_attack = attack_rate_per_hour * attack_duration_hours

# Print the result
print(f"Total log events ingested during the 3 hours of the attack: {int(total_events_during_attack)}")


OUTPUT:
Total log events ingested during the 3 hours of the attack: 167400


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
To calculate the total extra cost, we follow these steps:

1.  **Identify the total hours spent:** 65 hours.
2.  **Identify the retainer limit:** 40 hours.
3.  **Calculate the billable hours:** Subtract the retainer hours from the total hours spent (65 - 40).
4.  **Calculate the cost:** Multiply the billable hours by the hourly rate ($350).

Here is the Python code to perform this calculation:

```python
# Define the variables
total_hours_spent = 65
retainer_hours = 40
hourly_rate = 350

# Calculate billable hours
billable_hours = total_hours_spent - retainer_hours

# Calculate total extra cost
total_extra_cost = billable_hours * hourly_rate

# Print the result
print(f"The total extra cost billed for the incident is: ${total_extra_cost}")
```

------------------------------------------------------------
EXTRACTED CODE AND SAFE EXECUTION RESULT:
CODE:
# Define the variables
total_hours_spent = 65
retainer_hours = 40
hourly_rate = 350

# Calculate billable hours
billable_hours = total_hours_spent - retainer_hours

# Calculate total extra cost
total_extra_cost = billable_hours * hourly_rate

# Print the result
print(f"The total extra cost billed for the incident is: ${total_extra_cost}")


OUTPUT:
The total extra cost billed for the incident is: $8750


============================================================
SAFETY CHECK DEMO
============================================================

Reads a secret file:
Code rejected: Name not allowed: open

Runs a shell command:
Code rejected: Import not allowed: os

Climbs to hidden objects:
Code rejected: Attribute not allowed: __subclasses__

Never ends:
Code stopped after 5 seconds.

"""