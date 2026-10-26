"""
Title: Program of Thought Prompting (OpenRouter)
Program-of-Thought prompting asks the model to solve a problem by
writing Python code instead of reasoning in plain text, so the actual
arithmetic is done by a real interpreter, not guessed by the model.

This version has the model assign the final answer to a variable
called `answer`, then reads that variable directly out of the
executed code's namespace instead of capturing printed output.

uv run python src/12_program_of_thought_openrouter.py
"""

import re

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_and_run_code(response: str) -> str:
    """Extract Python code from the response and execute it, reading the
    final answer from a variable instead of capturing printed output."""

    # Find the fenced code block in the model's response
    code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
    if not code_match:
        return f"No code found. Raw response:\n{response}"

    code = code_match.group(1)

    # Execute the extracted code and read the answer variable directly
    try:
        namespace = {}
        exec(code, {"__builtins__": __builtins__}, namespace)

        if "answer" not in namespace:
            return f"No 'answer' variable found.\n\nCODE:\n{code}"

        answer = namespace["answer"]
        return f"CODE:\n{code}\nANSWER: {answer}"
    except Exception as e:
        return f"Error executing code: {e}\n\nCODE:\n{code}"


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

A permissions/intents classifier processes 25,000 APKs per day.
        It flags 8% as suspicious. Of the flagged APKs, 22% are confirmed
        malicious after manual review. How many APKs are confirmed
        malicious per day?

------------------------------------------------------------
ORIGINAL MODEL RESPONSE:
To solve this problem, we need to follow these steps:

1. Calculate the number of APKs flagged as suspicious per day.
2. Calculate the number of APKs confirmed as malicious from the flagged ones.

Let's break it down:

1. **Calculate the number of flagged APKs:**
   - 8% of 25,000 APKs are flagged as suspicious.
   
2. **Calculate the number of confirmed malicious APKs:**
   - 22% of the flagged APKs are confirmed as malicious.

Now, let's write the Python code to compute this:

```python
# Total number of APKs processed per day
total_apks = 25000

# Percentage of APKs flagged as suspicious
percentage_flagged = 8 / 100

# Percentage of flagged APKs that are confirmed malicious
percentage_malicious = 22 / 100

# Calculate the number of flagged APKs
flagged_apks = total_apks * percentage_flagged

# Calculate the number of confirmed malicious APKs
confirmed_malicious_apks = flagged_apks * percentage_malicious

# Assign the final answer to the variable `answer`
answer = confirmed_malicious_apks
```

------------------------------------------------------------
EXTRACTED CODE AND EXECUTION RESULT:
CODE:
# Total number of APKs processed per day
total_apks = 25000

# Percentage of APKs flagged as suspicious
percentage_flagged = 8 / 100

# Percentage of flagged APKs that are confirmed malicious
percentage_malicious = 22 / 100

# Calculate the number of flagged APKs
flagged_apks = total_apks * percentage_flagged

# Calculate the number of confirmed malicious APKs
confirmed_malicious_apks = flagged_apks * percentage_malicious

# Assign the final answer to the variable `answer`
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
To solve this problem, we need to calculate the total time required to analyze all 18,000 APKs using 6 parallel workers. Since the scanner takes 45 seconds to analyze one APK and it runs on 6 parallel workers, each worker can analyze one APK simultaneously. Therefore, the time taken to analyze all APKs will be equal to the time taken by one worker to analyze one APK (since they are working in parallel).

Here's the step-by-step approach:
1. Calculate the total time required for one worker to analyze all APKs.
2. Since there are 6 parallel workers, the total time remains the same as the time taken by one worker.
3. Convert the time from seconds to minutes.

Now, let's write the Python code:

```python
# Number of APKs
num_apks = 18000

# Time to analyze one APK in seconds
time_per_apk_seconds = 45

# Number of parallel workers
num_workers = 6

# Total time in seconds for one worker to analyze all APKs
total_time_seconds = num_apks * time_per_apk_seconds / num_workers

# Convert total time to minutes
total_time_minutes = total_time_seconds / 60

answer = total_time_minutes
```

------------------------------------------------------------
EXTRACTED CODE AND EXECUTION RESULT:
CODE:
# Number of APKs
num_apks = 18000

# Time to analyze one APK in seconds
time_per_apk_seconds = 45

# Number of parallel workers
num_workers = 6

# Total time in seconds for one worker to analyze all APKs
total_time_seconds = num_apks * time_per_apk_seconds / num_workers

# Convert total time to minutes
total_time_minutes = total_time_seconds / 60

answer = total_time_minutes

ANSWER: 2250.0

"""