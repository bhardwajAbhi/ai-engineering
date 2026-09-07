"""
Title: Chain of Thought Prompting (Google Generative AI)
uv run python src/07_chain_of_thought_google.py
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


def main():
    load_dotenv()

    # Initialize the model
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Chain-of-Thought prompt
    cot_prompt = ChatPromptTemplate.from_template("""
        You are an experienced cybersecurity analyst.

        Solve the following cybersecurity problem using clear and verifiable calculation steps.

        Problem:

        {problem}

        Instructions:

        1. Identify the given values.
        2. Determine the order of calculations.
        3. Show the formula used in each step.
        4. Perform each calculation sequentially.
        5. Clearly state the final answer.

    """)

    # Create the chain
    chain = cot_prompt | llm | parser

    # Sample problems
    problems = [
        """
        A phishing campaign sends 2,000 emails. The email security gateway
        blocks 80% of them. Of the emails that reach employees, 15% of users
        click the malicious link. Of those who click, 20% enter their passwords.
        Multi-factor authentication prevents account access for 75% of those
        users. How many accounts could the attacker potentially access?
        """,
        """
        An organisation has 500 computers. Security patches have been
        installed on 82% of them. Of the unpatched computers, 40% are
        accessible from the internet. Of those internet-accessible computers,
        25% contain a critical vulnerability. How many computers are both
        internet-accessible and critically vulnerable?
        """,
        """
        A security system monitors 250 devices. Each device generates
        20 MB of security logs per day. Logs must be retained for 90 days.
        Compression reduces the required storage by 40%. Using
        1 GB = 1,000 MB, how much storage is required after compression?
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
A phishing campaign sends 2,000 emails. The email security gateway
        blocks 80% of them. Of the emails that reach employees, 15% of users
        click the malicious link. Of those who click, 20% enter their passwords.
        Multi-factor authentication prevents account access for 75% of those
        users. How many accounts could the attacker potentially access?

Solution:
To determine the number of accounts the attacker could potentially access, we will break the problem down into sequential steps based on the flow of the phishing campaign.

### 1. Identify the Given Values
*   **Total emails sent:** 2,000
*   **Gateway block rate:** 80%
*   **User click rate (of those delivered):** 15%
*   **Password entry rate (of those who clicked):** 20%
*   **MFA prevention rate (of those who entered passwords):** 75%

---

### 2. Order of Calculations
1.  Calculate the number of emails that bypass the gateway.
2.  Calculate the number of users who click the link.
3.  Calculate the number of users who enter their passwords.
4.  Calculate the number of accounts compromised after MFA.

---

### 3. Calculations

**Step 1: Emails that reach employees**
*   *Formula:* Total Emails × (1 - Block Rate)
*   *Calculation:* $2,000 \times (1 - 0.80) = 2,000 \times 0.20 = 400$ emails delivered.

**Step 2: Users who click the link**
*   *Formula:* Delivered Emails × Click Rate
*   *Calculation:* $400 \times 0.15 = 60$ users clicked.

**Step 3: Users who enter their passwords**
*   *Formula:* Clicked Emails × Password Entry Rate
*   *Calculation:* $60 \times 0.20 = 12$ users entered passwords.

**Step 4: Accounts compromised after MFA**
*   *Formula:* Password Entries × (1 - MFA Prevention Rate)
*   *Calculation:* $12 \times (1 - 0.75) = 12 \times 0.25 = 3$ accounts.

---

### 4. Final Answer
The attacker could potentially access **3** accounts.
============================================================

Problem:
An organisation has 500 computers. Security patches have been
        installed on 82% of them. Of the unpatched computers, 40% are
        accessible from the internet. Of those internet-accessible computers,
        25% contain a critical vulnerability. How many computers are both
        internet-accessible and critically vulnerable?

Solution:
As an experienced cybersecurity analyst, I have broken down the problem into logical steps to determine the number of high-risk assets within the organization.

### 1. Identify the Given Values
*   **Total computers ($T$):** 500
*   **Patch rate:** 82%
*   **Unpatched rate:** 100% - 82% = 18%
*   **Internet-accessible rate (of unpatched):** 40%
*   **Critical vulnerability rate (of internet-accessible):** 25%

---

### 2. Determine the Order of Calculations
1.  Calculate the total number of **unpatched computers**.
2.  Calculate the number of **unpatched computers that are internet-accessible**.
3.  Calculate the number of **internet-accessible computers that are critically vulnerable**.

---

### 3. Calculations

**Step 1: Calculate the number of unpatched computers ($U$)**
*   **Formula:** $U = T \times (1 - \text{Patch Rate})$
*   **Calculation:** $500 \times 0.18 = 90$
*   *Result:* There are 90 unpatched computers.

**Step 2: Calculate the number of internet-accessible, unpatched computers ($A$)**
*   **Formula:** $A = U \times \text{Internet-accessible Rate}$
*   **Calculation:** $90 \times 0.40 = 36$
*   *Result:* There are 36 unpatched computers accessible from the internet.

**Step 3: Calculate the number of internet-accessible, critically vulnerable computers ($V$)**
*   **Formula:** $V = A \times \text{Critical Vulnerability Rate}$
*   **Calculation:** $36 \times 0.25 = 9$
*   *Result:* There are 9 computers that meet both criteria.

---

### 4. Final Answer
There are **9** computers that are both internet-accessible and critically vulnerable.
============================================================

Problem:
A security system monitors 250 devices. Each device generates
        20 MB of security logs per day. Logs must be retained for 90 days.
        Compression reduces the required storage by 40%. Using
        1 GB = 1,000 MB, how much storage is required after compression?

Solution:
To determine the total storage required for the security logs, we will follow a structured calculation process.

### 1. Identify the Given Values
*   **Number of devices:** 250
*   **Logs per device per day:** 20 MB
*   **Retention period:** 90 days
*   **Compression rate:** 40% reduction (meaning 60% of the original size remains)
*   **Conversion factor:** 1 GB = 1,000 MB

---

### 2. Determine the Order of Calculations
1.  Calculate the total daily log volume for all devices.
2.  Calculate the total volume for the 90-day retention period.
3.  Apply the compression factor to find the compressed size in MB.
4.  Convert the final result from MB to GB.

---

### 3. Formulas and Sequential Calculations

**Step 1: Calculate total daily log volume**
*   *Formula:* (Number of devices) × (Logs per device per day)
*   *Calculation:* 250 devices × 20 MB/device = **5,000 MB/day**

**Step 2: Calculate total volume for 90 days**
*   *Formula:* (Daily volume) × (Retention period)
*   *Calculation:* 5,000 MB/day × 90 days = **450,000 MB**

**Step 3: Apply compression**
*   *Note:* A 40% reduction means we retain 60% (100% - 40% = 60%) of the original data.
*   *Formula:* (Total volume) × (1 - Compression rate)
*   *Calculation:* 450,000 MB × 0.60 = **270,000 MB**

**Step 4: Convert to GB**
*   *Formula:* (Total compressed MB) / 1,000
*   *Calculation:* 270,000 MB / 1,000 = **270 GB**

---

### 4. Final Answer
The total storage required after compression is **270 GB**.
============================================================





"""