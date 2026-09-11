"""
Title: Least-to-Most Prompting (Google Generative AI)

Least-to-most prompting breaks a complex problem into a sequence of
simpler subproblems, then solves them in order, using earlier
solutions to help solve the later ones.

uv run python src/11_least_to_most_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Step 1: Decompose the problem into subproblems
    decompose_prompt = ChatPromptTemplate.from_template("""
        You are an experienced cybersecurity analyst.

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
        A SOC has 3 shifts: Morning (12 analysts), Afternoon (10 analysts),
        and Night (6 analysts). Morning shift is adding 25% more analysts,
        Afternoon is adding 4 new analysts, and Night is losing 2 analysts
        due to attrition. After these changes, what percentage of the
        total SOC staff will be on the Night shift?
        """,

        """
        A vulnerability scanner found 800 findings. 40% are classified as
        Critical, 35% as High, and the rest as Medium. Critical findings
        take 3 hours each to remediate, High findings take 1.5 hours each,
        and Medium findings take 30 minutes each. What is the total
        remediation time in hours?
        """,

        """
        An organisation has 1,200 endpoints. 65% are already enrolled in
        EDR. IT can enrol 15 unenrolled endpoints per day. If leadership
        wants 90% of all endpoints enrolled, how many days will it take
        from today?
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

A SOC has 3 shifts: Morning (12 analysts), Afternoon (10 analysts),
        and Night (6 analysts). Morning shift is adding 25% more analysts,
        Afternoon is adding 4 new analysts, and Night is losing 2 analysts
        due to attrition. After these changes, what percentage of the
        total SOC staff will be on the Night shift?

------------------------------------------------------------
SUBPROBLEMS:
As an experienced cybersecurity analyst, I approach this resource allocation problem by breaking it down into discrete, verifiable steps to ensure accuracy in the final calculation.

**SUBPROBLEMS (from simplest to most complex):**

1.  **Calculate the new Morning shift headcount:** Determine the result of adding 25% to the current 12 analysts (12 * 1.25).
2.  **Calculate the new Afternoon shift headcount:** Determine the result of adding 4 analysts to the current 10 (10 + 4).
3.  **Calculate the new Night shift headcount:** Determine the result of subtracting 2 analysts from the current 6 (6 - 2).
4.  **Calculate the total SOC staff headcount:** Sum the results of the three previous subproblems to find the new total workforce.
5.  **Calculate the Night shift percentage:** Divide the new Night shift headcount (from subproblem 3) by the new total SOC staff (from subproblem 4) and multiply by 100 to get the final percentage.

------------------------------------------------------------
STEP-BY-STEP SOLUTION:
Here are the step-by-step solutions to the subproblems:

**1. Calculate the new Morning shift headcount:**
Current headcount is 12. Adding 25% (12 * 0.25 = 3) results in:
12 + 3 = **15 analysts**

**2. Calculate the new Afternoon shift headcount:**
Current headcount is 10. Adding 4 new analysts results in:
10 + 4 = **14 analysts**

**3. Calculate the new Night shift headcount:**
Current headcount is 6. Losing 2 analysts results in:
6 - 2 = **4 analysts**

**4. Calculate the total SOC staff headcount:**
Summing the new totals from the three shifts:
15 (Morning) + 14 (Afternoon) + 4 (Night) = **33 total analysts**

**5. Calculate the Night shift percentage:**
Divide the new Night shift headcount (4) by the new total SOC staff (33) and multiply by 100:
(4 / 33) * 100 ≈ 0.1212 * 100 = **12.12%**

**FINAL ANSWER:**
After the staffing changes, the Night shift will comprise approximately **12.12%** of the total SOC staff.

============================================================
PROBLEM 2
============================================================

A vulnerability scanner found 800 findings. 40% are classified as
        Critical, 35% as High, and the rest as Medium. Critical findings
        take 3 hours each to remediate, High findings take 1.5 hours each,
        and Medium findings take 30 minutes each. What is the total
        remediation time in hours?

------------------------------------------------------------
SUBPROBLEMS:
As an experienced cybersecurity analyst, I have broken down this remediation estimation task into logical, sequential steps to ensure accuracy and scalability.

**SUBPROBLEMS (from simplest to most complex):**

1.  **Calculate the quantity of findings per severity level:** Determine the exact number of findings for Critical, High, and Medium categories based on the total count of 800 and the provided percentages.
2.  **Convert all remediation time units to a standard format:** Ensure all time estimates are in hours (e.g., converting the 30-minute Medium remediation time into 0.5 hours).
3.  **Calculate the total remediation time for each severity category:** Multiply the quantity of findings (from Subproblem 1) by the time required per finding (from Subproblem 2) for each category.
4.  **Aggregate the total remediation time:** Sum the results from Subproblem 3 to arrive at the final total project duration in hours.

------------------------------------------------------------
STEP-BY-STEP SOLUTION:
To determine the total remediation time, we will follow the subproblems as outlined:

### 1. Calculate the quantity of findings per severity level
*   **Total findings:** 800
*   **Critical (40%):** $800 \times 0.40 = 320$ findings
*   **High (35%):** $800 \times 0.35 = 280$ findings
*   **Medium (The rest):** $100\% - (40\% + 35\%) = 25\%$
    *   $800 \times 0.25 = 200$ findings

### 2. Convert all remediation time units to a standard format (hours)
*   **Critical:** 3 hours
*   **High:** 1.5 hours
*   **Medium:** 30 minutes = 0.5 hours

### 3. Calculate the total remediation time for each severity category
*   **Critical:** $320 \text{ findings} \times 3 \text{ hours/finding} = 960 \text{ hours}$
*   **High:** $280 \text{ findings} \times 1.5 \text{ hours/finding} = 420 \text{ hours}$
*   **Medium:** $200 \text{ findings} \times 0.5 \text{ hours/finding} = 100 \text{ hours}$

### 4. Aggregate the total remediation time
*   **Total Time:** $960 + 420 + 100 = 1,480 \text{ hours}$

**Final Answer:**
The total remediation time required for all 800 findings is **1,480 hours**.

============================================================
PROBLEM 3
============================================================

An organisation has 1,200 endpoints. 65% are already enrolled in
        EDR. IT can enrol 15 unenrolled endpoints per day. If leadership
        wants 90% of all endpoints enrolled, how many days will it take
        from today?

------------------------------------------------------------
SUBPROBLEMS:
As an experienced cybersecurity analyst, I have broken this deployment project down into logical, sequential steps to determine the timeline for reaching your 90% coverage goal.

**SUBPROBLEMS (from simplest to most complex):**

1.  **Calculate the current number of enrolled endpoints:** Determine the exact count of the 1,200 endpoints that are already protected (65% of 1,200).
2.  **Calculate the target number of enrolled endpoints:** Determine the exact count required to reach the 90% coverage goal (90% of 1,200).
3.  **Calculate the enrollment gap:** Subtract the current number of enrolled endpoints (from Subproblem 1) from the target number (from Subproblem 2) to find the total number of endpoints that still need to be onboarded.
4.  **Calculate the time to completion:** Divide the enrollment gap (from Subproblem 3) by the daily enrollment capacity (15 endpoints per day) to determine the total number of days required.

***

**Executive Summary of the Calculation:**
*   **Current Enrolled:** 780 endpoints (1,200 * 0.65)
*   **Target Enrolled:** 1,080 endpoints (1,200 * 0.90)
*   **Enrollment Gap:** 300 endpoints (1,080 - 780)
*   **Days Required:** **20 days** (300 / 15)

------------------------------------------------------------
STEP-BY-STEP SOLUTION:
To determine the timeline for reaching the 90% EDR coverage goal, we will solve the subproblems sequentially as outlined:

### Step 1: Calculate the current number of enrolled endpoints
We need to find 65% of the total 1,200 endpoints.
*   Calculation: $1,200 \times 0.65 = 780$
*   **Result:** There are currently **780** endpoints enrolled.

### Step 2: Calculate the target number of enrolled endpoints
We need to find 90% of the total 1,200 endpoints to meet the leadership goal.
*   Calculation: $1,200 \times 0.90 = 1,080$
*   **Result:** The target is **1,080** enrolled endpoints.

### Step 3: Calculate the enrollment gap
We subtract the current number of enrolled endpoints (from Step 1) from the target number (from Step 2) to identify how many more endpoints must be onboarded.
*   Calculation: $1,080 - 780 = 300$
*   **Result:** There is an enrollment gap of **300** endpoints.

### Step 4: Calculate the time to completion
We divide the enrollment gap (from Step 3) by the IT team's daily capacity of 15 endpoints per day.
*   Calculation: $300 / 15 = 20$
*   **Result:** It will take **20 days** to reach the goal.

***

### Final Answer:
Based on the current enrollment of 780 endpoints and a target of 1,080, the IT team must enroll 300 additional endpoints. At a rate of 15 endpoints per day, it will take **20 days** from today to reach the 90% coverage goal.




"""