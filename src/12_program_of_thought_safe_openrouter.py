"""
Title: Program of Thought Prompting, Safe Execution (OpenRouter)
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

The model assigns the final answer to a variable called `answer`.
We add one fixed line after the model's code that prints repr(answer),
then read that value back from the separate process.

This is a basic fix for learning, not a full sandbox. Real systems add
containers, OS-level limits or dedicated sandboxes on top.

uv run python src/12_program_of_thought_safe_openrouter.py
"""

import ast
import re
import subprocess
import sys
import tempfile

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
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

# Our own line, added after the model's code (it is not checked,
# because we wrote it). It prints the value of `answer`, or a marker
# if the model forgot to set it.
ANSWER_FOOTER = """
try:
    print(repr(answer))
except NameError:
    print("<<NO_ANSWER>>")
"""


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
    """Check the code, run it in a separate, limited Python process and
    return the value of its `answer` variable."""

    try:
        check_code(code)
    except (SyntaxError, ValueError) as e:
        return f"Code rejected: {e}"

    # An empty temporary folder as the working directory
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code + ANSWER_FOOTER],
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

    # The answer is the last line that our footer printed
    lines = result.stdout.strip().splitlines()
    last_line = lines[-1] if lines else ""

    if last_line == "<<NO_ANSWER>>":
        return "No 'answer' variable found."

    return last_line


def extract_and_run_code(response: str) -> str:
    """Extract Python code from the response and run it safely."""

    # Find the fenced code block in the model's response
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        return f"No code found. Raw response:\n{response}"

    code = code_match.group(1)
    answer = run_code_safely(code)
    return f"CODE:\n{code}\nANSWER: {answer}"


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Program-of-Thought prompt
    pot_prompt = ChatPromptTemplate.from_template("""
        You are an Android application security engineer.

        Solve this problem by writing Python code.
        Think through the problem, then write code that computes the answer.
        Put your code in a ```python``` block.
        Use only plain Python and, if needed, the math module.

        PROBLEM: {problem}

        First, explain your approach briefly.
        Then write the Python code.
        The code should assign the final answer to a variable called
        `answer`. Do not print anything.
    """)

    # Create the chain
    chain = pot_prompt | llm | parser

    # Sample security problems
    problems = [
        """
        A permissions/intents classifier processes 25,000 APKs per day.
        It flags 8% as suspicious. Of the flagged APKs, 22% are confirmed
        malicious after manual review. How many APKs are confirmed
        malicious per day?
        """,
        """
        An app store's automated scanner takes 45 seconds to analyse one
        APK. There are 18,000 APKs in today's queue. If the scanner runs
        on 6 parallel workers, how many minutes will it take to clear
        the queue?
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
        "Reads a secret file": 'answer = open(".env").read()',
        "Runs a shell command": 'import os\nanswer = os.system("whoami")',
        "Climbs to hidden objects": "answer = ().__class__.__bases__[0].__subclasses__()",
        "Never ends": "while True:\n    pass",
        "Forgets the answer variable": "result = 2 + 2",
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

A permissions/intents classifier processes 25,000 APKs per day.
        It flags 8% as suspicious. Of the flagged APKs, 22% are confirmed
        malicious after manual review. How many APKs are confirmed
        malicious per day?

------------------------------------------------------------
ORIGINAL MODEL RESPONSE:
To solve this problem, we need to follow these steps:

1. Calculate the number of APKs flagged as suspicious per day.
2. Calculate the number of flagged APKs that are confirmed malicious after manual review.

Let's break it down:

1. **Calculate the number of flagged APKs:**
   - Total APKs processed per day = 25,000
   - Percentage flagged as suspicious = 8%
   - Number of flagged APKs = Total APKs * (Percentage flagged / 100)

2. **Calculate the number of confirmed malicious APKs:**
   - Percentage of flagged APKs that are confirmed malicious = 22%
   - Number of confirmed malicious APKs = Number of flagged APKs * (Percentage confirmed malicious / 100)

Now, let's write the Python code to perform these calculations and assign the final answer to a variable called `answer`.

```python
# Define the given values
total_apks_per_day = 25000
percentage_flagged_suspicious = 8
percentage_confirmed_malicious = 22

# Calculate the number of flagged APKs
flagged_apks = total_apks_per_day * (percentage_flagged_suspicious / 100)

# Calculate the number of confirmed malicious APKs
confirmed_malicious_apks = flagged_apks * (percentage_confirmed_malicious / 100)

# Assign the final answer to the variable 'answer'
answer = confirmed_malicious_apks
```

------------------------------------------------------------
EXTRACTED CODE AND SAFE EXECUTION RESULT:
CODE:
# Define the given values
total_apks_per_day = 25000
percentage_flagged_suspicious = 8
percentage_confirmed_malicious = 22

# Calculate the number of flagged APKs
flagged_apks = total_apks_per_day * (percentage_flagged_suspicious / 100)

# Calculate the number of confirmed malicious APKs
confirmed_malicious_apks = flagged_apks * (percentage_confirmed_malicious / 100)

# Assign the final answer to the variable 'answer'
answer = confirmed_malicious_apks

ANSWER: 440.0

============================================================
PROBLEM 2
============================================================

An app store's automated scanner takes 45 seconds to analyse one
        APK. There are 18,000 APKs in today's queue. If the scanner runs
        on 6 parallel workers, how many minutes will it take to clear
        the queue?

------------------------------------------------------------
ORIGINAL MODEL RESPONSE:
To solve this problem, we need to determine the total time required to analyze all 18,000 APKs using 6 parallel workers. Since the analysis of each APK takes 45 seconds and there are 6 workers running in parallel, we can calculate the time it takes for one worker to complete its share of the work and then scale it up to account for all workers.

Here's the step-by-step approach:
1. Calculate the number of APKs each worker will handle.
2. Determine the time it takes for one worker to process its share of APKs.
3. Since all workers run in parallel, the total time is determined by the longest time taken by any single worker.

Let's write the Python code to compute this:

```python
# Number of APKs
total_apks = 18000

# Time to analyze one APK (in seconds)
time_per_apk = 45

# Number of parallel workers
num_workers = 6

# Calculate the number of APKs each worker will handle
apks_per_worker = total_apks / num_workers

# Calculate the time it takes for one worker to process its share of APKs
time_per_worker = apks_per_worker * time_per_apk

# Since all workers run in parallel, the total time is the same as the time taken by one worker
total_time_seconds = time_per_worker

# Convert the total time from seconds to minutes
answer = total_time_seconds / 60
```

------------------------------------------------------------
EXTRACTED CODE AND SAFE EXECUTION RESULT:
CODE:
# Number of APKs
total_apks = 18000

# Time to analyze one APK (in seconds)
time_per_apk = 45

# Number of parallel workers
num_workers = 6

# Calculate the number of APKs each worker will handle
apks_per_worker = total_apks / num_workers

# Calculate the time it takes for one worker to process its share of APKs
time_per_worker = apks_per_worker * time_per_apk

# Since all workers run in parallel, the total time is the same as the time taken by one worker
total_time_seconds = time_per_worker

# Convert the total time from seconds to minutes
answer = total_time_seconds / 60

ANSWER: 2250.0

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

Forgets the answer variable:
No 'answer' variable found.

"""