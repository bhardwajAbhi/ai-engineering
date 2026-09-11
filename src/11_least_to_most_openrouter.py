"""
Title: Least-to-Most Prompting (OpenRouter)

Least-to-most prompting breaks a complex problem into a sequence of
simpler subproblems, then solves them in order, using earlier
solutions to help solve the later ones.

uv run python src/11_least_to_most_openrouter.py
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

    # Step 1: Decompose the problem into subproblems
    decompose_prompt = ChatPromptTemplate.from_template("""
        You are an Android application security engineer.

        Break down this complex problem into smaller subproblems.
        List them from simplest to most complex.
        Each subproblem should be solvable on its own or using solutions
        from previous subproblems.

        PROBLEM: {problem}

        SUBPROBLEMS (from simplest to most complex):
        1.
    """)

    # Step 2: Solve the subproblems incrementally
    solve_prompt = ChatPromptTemplate.from_template("""
        Solve each subproblem in order, using solutions from previous
        subproblems when needed.

        ORIGINAL PROBLEM: {problem}

        SUBPROBLEMS:
        {subproblems}

        Solve each subproblem step by step, then combine for the final answer.

        SOLUTIONS:
    """)

    # The two chains that make up the least-to-most pipeline
    decompose_chain = decompose_prompt | llm | parser
    solve_chain = solve_prompt | llm | parser

    # Sample security problems
    problems = [
        """
        An app store review queue has 3 categories: Games (500 apps),
        Utilities (300 apps), and Finance (200 apps). Games submissions
        increase by 20%, Utilities gains 50 new apps, and Finance
        submissions drop by 10%. After these changes, what percentage
        of the total queue will be Finance apps?
        """,
        """
        A permissions/intents classifier scans 10,000 apps. 12% are
        flagged as suspicious. Of the flagged apps, 30% are confirmed
        malicious after manual review, and each confirmed malicious app
        takes 45 minutes to write a takedown report for. How many total
        hours are spent writing takedown reports?
        """,
        """
        A mobile security team must patch 400 apps for a newly disclosed
        WebView vulnerability. They can patch 25 apps per day, but every
        5th day is reserved for regression testing with no patches
        shipped that day. How many calendar days will it take to patch
        all 400 apps?
        """,
    ]

    # Invoke the least-to-most pipeline for each problem, one at a time
    for i, problem in enumerate(problems, 1):
        print(f"\n{'=' * 60}")
        print(f"PROBLEM {i}")
        print("=" * 60)
        print(f"\n{problem.strip()}\n")
        print("-" * 60)

        # Decompose the problem
        subproblems = decompose_chain.invoke({"problem": problem})
        print("SUBPROBLEMS:")
        print(subproblems)
        print("\n" + "-" * 60)

        # Solve the subproblems and combine into a final answer
        solution = solve_chain.invoke({"problem": problem, "subproblems": subproblems})
        print("STEP-BY-STEP SOLUTION:")
        print(solution)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------



============================================================
PROBLEM 1
============================================================

An app store review queue has 3 categories: Games (500 apps),
        Utilities (300 apps), and Finance (200 apps). Games submissions
        increase by 20%, Utilities gains 50 new apps, and Finance
        submissions drop by 10%. After these changes, what percentage
        of the total queue will be Finance apps?

------------------------------------------------------------
SUBPROBLEMS:
Sure, let's break down the problem into smaller subproblems, starting from the simplest to the most complex.

### SUBPROBLEMS

1. **Determine the initial number of apps in each category:**
   - Games: 500 apps
   - Utilities: 300 apps
   - Finance: 200 apps

2. **Calculate the new number of Games apps after a 20% increase:**
   - New number of Games apps = Initial number of Games apps + (20% of Initial number of Games apps)
   - New number of Games apps = 500 + (0.20 × 500) = 500 + 100 = 600 apps

3. **Calculate the new number of Utilities apps after gaining 50 new apps:**
   - New number of Utilities apps = Initial number of Utilities apps + 50
   - New number of Utilities apps = 300 + 50 = 350 apps

4. **Calculate the new number of Finance apps after a 10% decrease:**
   - New number of Finance apps = Initial number of Finance apps - (10% of Initial number of Finance apps)
   - New number of Finance apps = 200 - (0.10 × 200) = 200 - 20 = 180 apps

5. **Calculate the total number of apps in the queue after the changes:**
   - Total number of apps = New number of Games apps + New number of Utilities apps + New number of Finance apps
   - Total number of apps = 600 + 350 + 180 = 1130 apps

6. **Calculate the percentage of the total queue that will be Finance apps:**
   - Percentage of Finance apps = (New number of Finance apps / Total number of apps) × 100
   - Percentage of Finance apps = (180 / 1130) × 100 ≈ 15.84%

### Summary of Subproblems

1. Determine the initial number of apps in each category.
2. Calculate the new number of Games apps after a 20% increase.
3. Calculate the new number of Utilities apps after gaining 50 new apps.
4. Calculate the new number of Finance apps after a 10% decrease.
5. Calculate the total number of apps in the queue after the changes.
6. Calculate the percentage of the total queue that will be Finance apps.

By solving these subproblems step-by-step, you can determine the final percentage of Finance apps in the review queue.

------------------------------------------------------------
STEP-BY-STEP SOLUTION:
Sure, let's solve each subproblem step by step and then combine the results to find the final answer.

### SUBPROBLEMS

1. **Determine the initial number of apps in each category:**
   - Games: 500 apps
   - Utilities: 300 apps
   - Finance: 200 apps

2. **Calculate the new number of Games apps after a 20% increase:**
   - New number of Games apps = Initial number of Games apps + (20% of Initial number of Games apps)
   - New number of Games apps = 500 + (0.20 × 500) = 500 + 100 = 600 apps

3. **Calculate the new number of Utilities apps after gaining 50 new apps:**
   - New number of Utilities apps = Initial number of Utilities apps + 50
   - New number of Utilities apps = 300 + 50 = 350 apps

4. **Calculate the new number of Finance apps after a 10% decrease:**
   - New number of Finance apps = Initial number of Finance apps - (10% of Initial number of Finance apps)
   - New number of Finance apps = 200 - (0.10 × 200) = 200 - 20 = 180 apps

5. **Calculate the total number of apps in the queue after the changes:**
   - Total number of apps = New number of Games apps + New number of Utilities apps + New number of Finance apps
   - Total number of apps = 600 + 350 + 180 = 1130 apps

6. **Calculate the percentage of the total queue that will be Finance apps:**
   - Percentage of Finance apps = (New number of Finance apps / Total number of apps) × 100
   - Percentage of Finance apps = (180 / 1130) × 100 ≈ 15.84%

### Final Answer

The percentage of the total queue that will be Finance apps is approximately **15.84%**.

============================================================
PROBLEM 2
============================================================

A permissions/intents classifier scans 10,000 apps. 12% are
        flagged as suspicious. Of the flagged apps, 30% are confirmed
        malicious after manual review, and each confirmed malicious app
        takes 45 minutes to write a takedown report for. How many total
        hours are spent writing takedown reports?

------------------------------------------------------------
SUBPROBLEMS:
Sure, let's break down the problem into smaller subproblems, starting from the simplest to the most complex.

### Subproblems

1. **Calculate the number of apps flagged as suspicious:**
   - Given that 12% of the 10,000 apps are flagged as suspicious.
   - Formula: \( \text{Number of flagged apps} = 10,000 \times 0.12 \)

2. **Calculate the number of confirmed malicious apps:**
   - Given that 30% of the flagged apps are confirmed as malicious.
   - Formula: \( \text{Number of confirmed malicious apps} = \text{Number of flagged apps} \times 0.30 \)

3. **Calculate the total time spent writing takedown reports:**
   - Each confirmed malicious app takes 45 minutes to write a takedown report.
   - Formula: \( \text{Total time in minutes} = \text{Number of confirmed malicious apps} \times 45 \)
   - Convert the total time from minutes to hours: \( \text{Total time in hours} = \frac{\text{Total time in minutes}}{60} \)

### Detailed Steps

1. **Calculate the number of apps flagged as suspicious:**
   \[
   \text{Number of flagged apps} = 10,000 \times 0.12 = 1,200
   \]

2. **Calculate the number of confirmed malicious apps:**
   \[
   \text{Number of confirmed malicious apps} = 1,200 \times 0.30 = 360
   \]

3. **Calculate the total time spent writing takedown reports:**
   \[
   \text{Total time in minutes} = 360 \times 45 = 16,200 \text{ minutes}
   \]
   \[
   \text{Total time in hours} = \frac{16,200}{60} = 270 \text{ hours}
   \]

### Final Answer
The total number of hours spent writing takedown reports is \( 270 \) hours.

------------------------------------------------------------
STEP-BY-STEP SOLUTION:
Sure, let's solve each subproblem step by step and then combine the results to find the final answer.

### Subproblems

1. **Calculate the number of apps flagged as suspicious:**
   - Given that 12% of the 10,000 apps are flagged as suspicious.
   - Formula: \( \text{Number of flagged apps} = 10,000 \times 0.12 \)

2. **Calculate the number of confirmed malicious apps:**
   - Given that 30% of the flagged apps are confirmed as malicious.
   - Formula: \( \text{Number of confirmed malicious apps} = \text{Number of flagged apps} \times 0.30 \)

3. **Calculate the total time spent writing takedown reports:**
   - Each confirmed malicious app takes 45 minutes to write a takedown report.
   - Formula: \( \text{Total time in minutes} = \text{Number of confirmed malicious apps} \times 45 \)
   - Convert the total time from minutes to hours: \( \text{Total time in hours} = \frac{\text{Total time in minutes}}{60} \)

### Detailed Steps

#### Step 1: Calculate the number of apps flagged as suspicious
\[
\text{Number of flagged apps} = 10,000 \times 0.12 = 1,200
\]

#### Step 2: Calculate the number of confirmed malicious apps
\[
\text{Number of confirmed malicious apps} = 1,200 \times 0.30 = 360
\]

#### Step 3: Calculate the total time spent writing takedown reports
\[
\text{Total time in minutes} = 360 \times 45 = 16,200 \text{ minutes}
\]
\[
\text{Total time in hours} = \frac{16,200}{60} = 270 \text{ hours}
\]

### Final Answer
The total number of hours spent writing takedown reports is \( 270 \) hours.

============================================================
PROBLEM 3
============================================================

A mobile security team must patch 400 apps for a newly disclosed
        WebView vulnerability. They can patch 25 apps per day, but every
        5th day is reserved for regression testing with no patches
        shipped that day. How many calendar days will it take to patch
        all 400 apps?

------------------------------------------------------------
SUBPROBLEMS:
Sure, let's break down the problem into smaller subproblems, starting from the simplest to the most complex.

### Subproblem 1:
**Calculate how many apps can be patched in a single day.**

- **Solution:** The team can patch 25 apps per day.

### Subproblem 2:
**Determine the number of days required to patch all 400 apps without considering regression testing.**

- **Solution:** 
  - Total apps to patch: 400
  - Apps patched per day: 25
  - Days required = Total apps / Apps patched per day = 400 / 25 = 16 days

### Subproblem 3:
**Account for the regression testing days where no patches are shipped.**

- **Solution:** Every 5th day is reserved for regression testing.
  - In 16 days, there will be 3 regression testing days (since \( \lfloor \frac{16}{5} \rfloor = 3 \)).
  - Therefore, the actual number of days available for patching is 16 - 3 = 13 days.

### Subproblem 4:
**Calculate the total number of apps that can be patched in the available days.**

- **Solution:** 
  - Apps patched per day: 25
  - Available days: 13
  - Apps patched in 13 days = 25 * 13 = 325 apps

### Subproblem 5:
**Determine if additional days are needed to patch the remaining apps after accounting for regression testing days.**

- **Solution:** 
  - Remaining apps to patch = 400 - 325 = 75 apps
  - Days required to patch the remaining apps = 75 / 25 = 3 days
  - Since these days do not fall on a regression testing day, they are valid working days.

### Subproblem 6:
**Calculate the total number of calendar days required to patch all 400 apps.**

- **Solution:** 
  - Days required for patching (excluding regression testing): 13 days
  - Additional days required for remaining apps: 3 days
  - Total calendar days = 13 + 3 = 16 days

### Final Answer:
The total number of calendar days required to patch all 400 apps is **16 days**.

------------------------------------------------------------
STEP-BY-STEP SOLUTION:
Sure, let's solve each subproblem step by step and then combine the solutions to find the final answer.

### Subproblem 1:
**Calculate how many apps can be patched in a single day.**

- **Solution:** The team can patch 25 apps per day.

### Subproblem 2:
**Determine the number of days required to patch all 400 apps without considering regression testing.**

- **Solution:**
  - Total apps to patch: 400
  - Apps patched per day: 25
  - Days required = Total apps / Apps patched per day = 400 / 25 = 16 days

### Subproblem 3:
**Account for the regression testing days where no patches are shipped.**

- **Solution:**
  - Every 5th day is reserved for regression testing.
  - In 16 days, there will be 3 regression testing days (since \( \lfloor \frac{16}{5} \rfloor = 3 \)).
  - Therefore, the actual number of days available for patching is 16 - 3 = 13 days.

### Subproblem 4:
**Calculate the total number of apps that can be patched in the available days.**

- **Solution:**
  - Apps patched per day: 25
  - Available days: 13
  - Apps patched in 13 days = 25 * 13 = 325 apps

### Subproblem 5:
**Determine if additional days are needed to patch the remaining apps after accounting for regression testing days.**

- **Solution:**
  - Remaining apps to patch = 400 - 325 = 75 apps
  - Days required to patch the remaining apps = 75 / 25 = 3 days
  - Since these days do not fall on a regression testing day, they are valid working days.

### Subproblem 6:
**Calculate the total number of calendar days required to patch all 400 apps.**

- **Solution:**
  - Days required for patching (excluding regression testing): 13 days
  - Additional days required for remaining apps: 3 days
  - Total calendar days = 13 + 3 = 16 days

### Final Answer:
The total number of calendar days required to patch all 400 apps is **16 days**.



"""