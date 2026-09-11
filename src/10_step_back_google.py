"""
Title: Step-Back Prompting (Google Generative AI)
Step-back prompting asks the model to first work out the general
principles behind a question, then apply those principles to answer
the specific question.

uv run python src/10_step_back_google.py
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

    # Step 1: Extract high-level security principles
    step_back_prompt = ChatPromptTemplate.from_template("""
        You are an experienced cybersecurity engineer skilled at
        identifying underlying security principles.

        Given this specific question:

        {question}

        What are the key security principles, concepts, or frameworks
        needed to answer this correctly? List 2-3 fundamental principles,
        then briefly explain each.

        PRINCIPLES:
    """)

    # Step 2: Apply the principles to the specific question
    apply_prompt = ChatPromptTemplate.from_template("""
        Use the following principles to answer the specific question.

        PRINCIPLES:
        {principles}

        SPECIFIC QUESTION:
        {question}

        Apply the principles step-by-step to arrive at the answer.

        ANSWER:
    """)

    # The two chains that make up the step-back pipeline
    step_back_chain = step_back_prompt | llm | parser
    apply_chain = apply_prompt | llm | parser

    # Sample security questions
    questions = [
        """
        A web application's login endpoint accepts unlimited password
        attempts with no rate limiting or account lockout. What security
        risks does this expose, and how should it be fixed?
        """,

        """
        A company stores user passwords using unsalted MD5 hashes. Why is
        this insecure, and what should be used instead?
        """,

        """
        An internal API endpoint trusts a client-supplied user_id
        parameter to decide which records to return, without checking
        that it belongs to the authenticated user. What vulnerability
        class is this, and how should it be fixed?
        """,
    ]

    # Invoke the step-back pipeline for each question, one at a time
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 60}")
        print(f"QUESTION {i}")
        print("=" * 60)
        print(f"\n{question.strip()}\n")
        print("-" * 60)

        # Show the step-back process
        principles = step_back_chain.invoke({"question": question})
        print("STEP-BACK PRINCIPLES:")
        print(principles)
        print("\n" + "-" * 60)

        # Apply the principles to reach the final answer
        answer = apply_chain.invoke({"question": question, "principles": principles})
        print("FINAL ANSWER:")
        print(answer)


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------



============================================================
QUESTION 1
============================================================

A web application's login endpoint accepts unlimited password
        attempts with no rate limiting or account lockout. What security
        risks does this expose, and how should it be fixed?

------------------------------------------------------------
STEP-BACK PRINCIPLES:
To address the security risks of an unprotected login endpoint, you must apply the following fundamental cybersecurity principles:

### PRINCIPLES:

**1. Defense in Depth**
This principle dictates that security should not rely on a single control. In this scenario, relying solely on a password is a single point of failure. By implementing rate limiting, account lockouts, or multi-factor authentication (MFA), you create multiple layers of defense. If an attacker bypasses one control (e.g., by using a proxy to rotate IPs), the next layer (e.g., account lockout or CAPTCHA) prevents the successful exploitation of the vulnerability.

**2. Least Privilege / Fail-Safe Defaults**
This principle suggests that systems should be designed to be secure by default. An endpoint that allows unlimited attempts is "failing open" rather than "failing closed." A secure design should default to the most restrictive state—denying access after a threshold of failed attempts—to ensure that the system is protected even if the user or administrator fails to configure additional security settings.

**3. Threat Modeling (specifically regarding Bruteforce/Credential Stuffing)**
This concept involves identifying potential threats to an application's assets. By modeling the threat of automated attacks, an engineer recognizes that an exposed login endpoint is a high-value target for credential stuffing (using leaked passwords) and brute-force attacks. Applying this principle leads to the implementation of proactive controls like rate limiting, which specifically mitigates the "unlimited attempts" vector identified in the threat model.

------------------------------------------------------------
FINAL ANSWER:
To address the security risks of an unprotected login endpoint, we apply the provided principles as follows:

### 1. Threat Modeling (Identifying the Risks)
By applying **Threat Modeling**, we identify that an exposed login endpoint with no restrictions is a high-value target for automated attacks. Specifically, this configuration exposes the application to:
*   **Brute-Force Attacks:** Attackers can systematically guess passwords for a specific account without fear of being blocked.
*   **Credential Stuffing:** Attackers can use lists of leaked credentials from other breaches to gain unauthorized access to user accounts, as the system does not distinguish between a legitimate user and a bot.
*   **Account Enumeration:** Without rate limiting, attackers can rapidly test usernames to determine which ones exist in the system.

### 2. Least Privilege / Fail-Safe Defaults (The Design Philosophy)
The current state of the application is "failing open," as it allows unlimited attempts by default. To adhere to **Fail-Safe Defaults**, the system must be redesigned to be secure from the start. The default state should be restrictive:
*   The system must be configured to automatically deny access after a predefined threshold of failed attempts.
*   This ensures that even if an administrator forgets to add advanced security features, the system remains protected against high-volume automated traffic.

### 3. Defense in Depth (The Remediation Strategy)
To move beyond a single point of failure (the password), we must implement **Defense in Depth** by layering multiple security controls. Relying on one mechanism is insufficient; therefore, the following layers should be implemented:
*   **Rate Limiting:** Implement per-IP or per-account rate limiting to throttle the speed at which login attempts can be made, effectively neutralizing high-speed brute-force tools.
*   **Account Lockout Policies:** Temporarily lock accounts after a specific number of failed attempts to prevent persistent guessing.
*   **Multi-Factor Authentication (MFA):** Introduce a second factor (e.g., TOTP, SMS, or hardware keys). Even if an attacker successfully guesses a password, they cannot bypass the second layer of authentication.
*   **CAPTCHA/Bot Detection:** Integrate challenges to distinguish between human users and automated scripts, adding a layer that is difficult for simple bots to bypass.

**Conclusion:**
By applying these principles, the application shifts from an insecure, "fail-open" state to a robust, multi-layered architecture. The combination of **Threat Modeling** (identifying the attack vector), **Fail-Safe Defaults** (restricting access by default), and **Defense in Depth** (layering controls like MFA and rate limiting) ensures the login endpoint is resilient against automated credential-based attacks.

============================================================
QUESTION 2
============================================================

A company stores user passwords using unsalted MD5 hashes. Why is
        this insecure, and what should be used instead?

------------------------------------------------------------
STEP-BACK PRINCIPLES:
To answer this question effectively, you must apply the following fundamental cybersecurity principles:

### PRINCIPLES:

**1. Defense in Depth (Layered Security)**
This principle dictates that security should not rely on a single control. Using unsalted MD5 hashes creates a "single point of failure" where the compromise of the database leads to the immediate exposure of all passwords. Adding a unique "salt" to each password ensures that even if two users have the same password, their stored hashes will be different, preventing attackers from using precomputed tables to crack multiple accounts simultaneously.

**2. Cryptographic Agility and Strength (Algorithm Obsolescence)**
This principle emphasizes that cryptographic algorithms must be evaluated for their resistance to modern computational power. MD5 is cryptographically broken; it is susceptible to collision attacks and is computationally "fast," meaning an attacker can calculate billions of hashes per second. A secure system must use "slow", memory-hard, or CPU-intensive hashing algorithms (like Argon2, bcrypt, or scrypt) designed specifically for password storage to thwart brute-force and dictionary attacks.

**3. The Principle of Least Privilege (Data Minimization)**
This principle suggests that systems should only store the minimum amount of information necessary to perform a function. Storing passwords in a reversible or easily crackable format violates the duty of care to protect user data. By using a modern, salted, and iterated hashing function, the system minimizes the "blast radius" of a data breach, ensuring that even if the database is exfiltrated, the actual passwords remain computationally infeasible to recover.

------------------------------------------------------------
FINAL ANSWER:
To determine why storing passwords as unsalted MD5 hashes is insecure and to identify the appropriate alternative, we apply the provided cybersecurity principles as follows:

### 1. Defense in Depth (Layered Security)
Storing passwords as unsalted MD5 hashes creates a **single point of failure**. Because the hashes are unsalted, identical passwords result in identical hash values across the entire database. An attacker who gains access to the database can use precomputed tables (such as Rainbow Tables) to instantly map these hashes back to their original passwords. By failing to implement a unique "salt" for every user, the system lacks the necessary layer of defense to prevent mass-cracking of accounts, meaning the compromise of the database leads to the immediate exposure of all user credentials.

### 2. Cryptographic Agility and Strength (Algorithm Obsolescence)
MD5 is cryptographically broken and fundamentally unsuitable for password storage. It is a "fast" hashing algorithm, meaning modern hardware can compute billions of MD5 hashes per second, making it trivial for an attacker to perform high-speed brute-force or dictionary attacks. To adhere to the principle of cryptographic strength, the company must replace MD5 with "slow," memory-hard, or CPU-intensive algorithms. These algorithms are specifically designed to be computationally expensive, significantly increasing the time and resources required for an attacker to test a single password guess.

### 3. The Principle of Least Privilege (Data Minimization)
Storing passwords in an easily crackable format violates the duty of care to protect user data. By using an obsolete algorithm like MD5, the company maximizes the "blast radius" of a potential data breach. To minimize this risk, the system should store only the most secure representation of the password possible. By transitioning to a modern, salted, and iterated hashing function, the company ensures that even if the database is exfiltrated, the actual passwords remain computationally infeasible to recover, thereby minimizing the impact of the breach on the users.

### Conclusion: What should be used instead?
The company must immediately stop using MD5. Instead, they should implement a **modern, salted, and iterated hashing algorithm** specifically designed for password storage. Recommended industry standards include:
*   **Argon2id** (the current state-of-the-art, winner of the Password Hashing Competition)
*   **bcrypt** (a widely trusted, adaptive, and slow hashing algorithm)
*   **scrypt** (designed to be memory-intensive to thwart hardware-accelerated attacks)

By adopting these algorithms with a unique, cryptographically strong salt for every user, the company will align its security posture with modern best practices.

============================================================
QUESTION 3
============================================================

An internal API endpoint trusts a client-supplied user_id
        parameter to decide which records to return, without checking
        that it belongs to the authenticated user. What vulnerability
        class is this, and how should it be fixed?

------------------------------------------------------------
STEP-BACK PRINCIPLES:
To address the vulnerability described—which is a classic **Insecure Direct Object Reference (IDOR)**—the following fundamental security principles are essential:

### PRINCIPLES:

1.  **Principle of Least Privilege (PoLP)**
    *   **Explanation:** This principle dictates that a user or process should only have access to the specific information and resources necessary for its legitimate purpose. In this scenario, the API fails to enforce this because it allows a user to access data (via `user_id`) that falls outside their authorized scope. Implementing PoLP ensures that the system verifies not just *who* the user is, but *what* they are explicitly permitted to access.

2.  **Authorization (Access Control)**
    *   **Explanation:** Authorization is the process of verifying that an authenticated user has the required permissions to perform an action or access a specific resource. The vulnerability exists because the system confuses **Authentication** (verifying identity) with **Authorization** (verifying permissions). A robust security architecture must perform an explicit authorization check on every request, ensuring the `user_id` requested matches the identity established during the authentication phase (e.g., via a secure session token or JWT).

3.  **Defense in Depth**
    *   **Explanation:** This principle advocates for layering multiple security controls so that if one fails, others are in place to prevent a breach. Relying solely on a client-supplied parameter is a "single point of failure." A defense-in-depth approach would involve validating the user's identity server-side (e.g., pulling the `user_id` from the secure session/token rather than the request body) and implementing server-side access control lists (ACLs) to ensure the requested resource belongs to the authenticated user.

------------------------------------------------------------
FINAL ANSWER:
**ANSWER:**

The vulnerability described is a classic **Insecure Direct Object Reference (IDOR)**. This occurs when an application provides direct access to objects based on user-supplied input without performing adequate authorization checks.

To fix this vulnerability, the following principles must be applied:

**1. Apply the Principle of Least Privilege (PoLP)**
The system must be reconfigured so that a user’s access is restricted strictly to their own data. Instead of allowing the client to dictate which `user_id` to query, the API should enforce a policy where the system only retrieves records explicitly associated with the authenticated user’s identity. This ensures that the user cannot "reach" into other users' data, as their privilege scope is programmatically limited to their own account.

**2. Implement Robust Authorization (Access Control)**
The API currently confuses authentication with authorization. While the user may be authenticated, the system fails to verify if they are *authorized* to access the requested `user_id`. To fix this, the server must perform an explicit authorization check on every request. The system should compare the `user_id` requested in the API call against the identity verified during the authentication phase (e.g., the `user_id` stored within a secure, server-side session or a cryptographically signed JWT). If the requested ID does not match the authenticated user's ID, the request must be denied.

**3. Utilize Defense in Depth**
To prevent the client-supplied parameter from being a single point of failure, the application should move away from trusting user input for resource identification. A defense-in-depth approach involves:
*   **Server-Side Source of Truth:** Instead of accepting `user_id` from the request body or URL parameters, the API should extract the user's identity directly from the secure, server-side session token.
*   **Server-Side Access Control Lists (ACLs):** Implement a secondary layer of validation where the database query itself is scoped to the authenticated user (e.g., `SELECT * FROM records WHERE user_id = [authenticated_user_id]`). This ensures that even if a malicious user attempts to manipulate the request, the backend logic inherently prevents access to unauthorized records.



"""