"""
Title: Chain of Thought Prompting (OpenRouter)
uv run python src/07_chain_of_thought_openrouter.py
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Chain of Thought prompt
    cot_prompt = ChatPromptTemplate.from_template(
        """
        You are an Android application security analyst.

        Analyze the following Android app analysis problem using clear
        and simple steps.

        Problem:

        {problem}

        Instructions:

        1. Identify the important information.
        2. Analyze it step by step.
        3. Clearly state the final answer.
        """
    )

    # Create the chain
    chain = cot_prompt | llm | parser

    # Sample Android app analysis problems
    problems = [
        """
        During analysis of an Android app, 12 API calls were observed.
        9 calls accessed normal application data and 3 calls accessed
        location information.

        What percentage of the observed API calls accessed location information?
        """,

        """
        During analysis of an Android app, 10 network connections were observed.
        8 connections were made to the app's known backend servers and
        2 connections were made to an unknown external server.

        What percentage of the connections were made to the unknown server?
        """,

        """
        An Android app was observed accessing the camera 6 times and the
        microphone 4 times during a 10-minute test.

        How many total sensitive resource accesses were observed, and
        what percentage were camera accesses?
        """,
    ]

    # Invoke the chain for each problem
    for problem in problems:
        result = chain.invoke({"problem": problem})

        print(f"\nProblem:\n{problem.strip()}")
        print(f"\nSolution:\n{result.strip()}")
        print("="*60)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------



Problem:
During analysis of an Android app, 12 API calls were observed.
        9 calls accessed normal application data and 3 calls accessed
        location information.

        What percentage of the observed API calls accessed location information?

Solution:
Sure, let's break down the problem step by step.

### Step 1: Identify the Important Information

- Total number of API calls observed: 12
- Number of API calls accessing normal application data: 9
- Number of API calls accessing location information: 3

### Step 2: Analyze It Step by Step

1. **Determine the total number of API calls**: We know this is 12.
2. **Identify the number of API calls that accessed location information**: This is given as 3.
3. **Calculate the percentage**:
   - The formula to calculate the percentage is: \(\text{Percentage} = \left( \frac{\text{Number of API calls accessing location information}}{\text{Total number of API calls}} \right) \times 100\)
   - Plugging in the numbers: \(\text{Percentage} = \left( \frac{3}{12} \right) \times 100\)

### Step 3: Clearly State the Final Answer

Now, let's do the calculation:

\[
\text{Percentage} = \left( \frac{3}{12} \right) \times 100 = 0.25 \times 100 = 25\%
\]

So, **25%** of the observed API calls accessed location information.
============================================================

Problem:
During analysis of an Android app, 10 network connections were observed.
        8 connections were made to the app's known backend servers and
        2 connections were made to an unknown external server.

        What percentage of the connections were made to the unknown server?

Solution:
Sure, let's break down the problem step by step.

### Step 1: Identify the Important Information

- Total number of network connections observed: 10
- Number of connections made to known backend servers: 8
- Number of connections made to an unknown external server: 2

### Step 2: Analyze It Step by Step

1. **Calculate the Percentage**:
   - We need to find out what percentage of the total connections were made to the unknown server.
   - The formula to calculate the percentage is:
     \[
     \text{Percentage} = \left( \frac{\text{Number of unknown server connections}}{\text{Total number of connections}} \right) \times 100
     \]

2. **Substitute the Values**:
   - Number of unknown server connections = 2
   - Total number of connections = 10
   - So, the calculation becomes:
     \[
     \text{Percentage} = \left( \frac{2}{10} \right) \times 100
     \]

3. **Perform the Calculation**:
   - Simplify the fraction first:
     \[
     \frac{2}{10} = 0.2
     \]
   - Then multiply by 100 to get the percentage:
     \[
     0.2 \times 100 = 20\%
     \]

### Step 3: Clearly State the Final Answer

The percentage of connections made to the unknown server is **20%**.
============================================================

Problem:
An Android app was observed accessing the camera 6 times and the
        microphone 4 times during a 10-minute test.

        How many total sensitive resource accesses were observed, and
        what percentage were camera accesses?

Solution:
Sure, let's break down the problem step by step.

### Step 1: Identify the Important Information

- The app accessed the camera 6 times.
- The app accessed the microphone 4 times.
- The total test duration was 10 minutes.

### Step 2: Analyze It Step by Step

1. **Calculate Total Sensitive Resource Accesses:**
   - Camera accesses: 6
   - Microphone accesses: 4
   - Total accesses = Camera accesses + Microphone accesses
   - Total accesses = 6 + 4 = 10

2. **Calculate Percentage of Camera Accesses:**
   - Total accesses = 10 (from the previous calculation)
   - Camera accesses = 6
   - Percentage of camera accesses = (Camera accesses / Total accesses) * 100%
   - Percentage of camera accesses = (6 / 10) * 100% = 60%

### Step 3: Clearly State the Final Answer

- The total number of sensitive resource accesses observed is **10**.
- The percentage of camera accesses is **60%**.

So, the final answer is:
- Total sensitive resource accesses: **10**
- Percentage of camera accesses: **60%**
============================================================



"""