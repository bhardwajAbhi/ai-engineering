"""
Title: Tree of Thought Prompting (Google Generative AI)
uv run python src/08_tree_of_thought_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    # A higher temperature encourages the three paths to genuinely differ
    # instead of converging on the same reasoning.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.7,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Tree-of-Thought prompt
    tot_prompt = ChatPromptTemplate.from_template("""
        You are an Android application security analyst.

        Analyse the following security decision by exploring MULTIPLE
        different response strategies, then compare them.

        Decision:

        {problem}

        Instructions:

        1. Generate 3 DIFFERENT response strategies.
        2. Label them as Path A, Path B, and Path C.
        3. For each path, show the reasoning and likely outcome.
        4. Evaluate which path carries the least risk and why.
        5. Recommend a final course of action based on the best path.

        Solution:

        PATH A:
        [First strategy]

        PATH B:
        [Second strategy]

        PATH C:
        [Third strategy]

        EVALUATION:
        [Compare the three paths]

        FINAL RECOMMENDATION:
        [Best course of action]
    """)

    # Create the chain
    chain = tot_prompt | llm | parser

    # Sample security decisions
    problems = [
        """
        A static analysis scan flags an Android app for requesting the
        SMS read permission with no visible feature that needs it. Should
        the review flag it as malicious, request developer justification,
        or run further dynamic analysis first?
        """,

        """
        A privacy audit finds that an Android app collects users' Aadhaar
        numbers and precise location during onboarding, but the privacy
        notice never mentions this collection and no separate consent is
        taken, as required under India's DPDP Act, 2023. Should the audit
        team demand immediate removal of the data field, require the
        developer to redesign the consent flow before re-review, or
        escalate the finding to the app store's compliance team?
        """,

        """
        A permissions/intents classifier flags a batch of 5,000 apps as
        suspicious with only 60% confidence. Should the team auto-quarantine
        all of them, manually review each one, or raise the confidence
        threshold and re-run the batch?
        """,
    ]

    # Invoke the chain for each problem
    for problem in problems:
        result = chain.invoke({"problem": problem})

        print(f"\nProblem:\n{problem.strip()}")
        print(f"\nSolution:\n{result.strip()}")
        print("=" * 60)


if __name__ == "__main__":
    main()



"""
-----------------------------------------
Output:
-----------------------------------------


Problem:
A static analysis scan flags an Android app for requesting the
        SMS read permission with no visible feature that needs it. Should
        the review flag it as malicious, request developer justification,
        or run further dynamic analysis first?

Solution:
### Analysis of Android SMS Permission Flag

As an Android security analyst, the detection of a sensitive permission like `READ_SMS` without a clear functional justification represents a potential violation of the Google Play Developer Policy regarding "User Data" and "Permissions."

---

### PATH A: Immediate Rejection (Flag as Malicious)
**Reasoning:** The `READ_SMS` permission is categorized as "Dangerous" by Android. If an app requests it without a clear, user-facing feature (like a messaging app or a banking app using OTP autofill), it is often indicative of credential harvesting or spyware. By rejecting immediately, the analyst enforces a "zero-trust" stance, preventing potentially harmful code from reaching users.
**Likely Outcome:** High developer frustration. If the developer has a legitimate, non-obvious use case (e.g., a security-focused app or a legacy feature), the app is blocked unnecessarily, leading to a lengthy appeal process.

### PATH B: Request Developer Justification (Interactive Review)
**Reasoning:** This path assumes the developer may have implemented a feature that the static analysis engine failed to identify. The analyst sends a formal inquiry requiring the developer to provide a "Permission Declaration Form" or a video walkthrough demonstrating why the permission is essential for core functionality.
**Likely Outcome:** The developer either provides a valid explanation (which can then be verified) or fails to respond/provides a vague explanation. This maintains a professional relationship while still upholding security standards.

### PATH C: Dynamic Analysis (Behavioral Sandbox)
**Reasoning:** Static analysis often misses obfuscated code or reflective permission requests. By running the APK in a sandboxed environment (Dynamic Analysis), the analyst can monitor API calls, network traffic, and file system access in real-time. If the app attempts to exfiltrate SMS data to a suspicious C2 (Command & Control) server, the malicious intent is confirmed objectively.
**Likely Outcome:** High confidence in the decision. If the app is benign, the analyst sees the permission is never actually utilized. If it is malicious, the analyst gains actionable evidence (IP addresses, exfiltrated data patterns) to justify a permanent ban.

---

### EVALUATION

| Criteria | Path A | Path B | Path C |
| :--- | :--- | :--- | :--- |
| **False Positive Risk** | High | Medium | Low |
| **Security Rigor** | High | Low | High |
| **Operational Effort** | Low | Medium | High |

*   **Path A** is too blunt; it risks punishing developers for poor documentation rather than actual malice.
*   **Path B** relies on the honesty of the developer, which is ineffective against bad actors.
*   **Path C** provides the most accurate assessment by validating *behavior* rather than just *intent*.

**Path C carries the least risk.** It minimizes false positives (by verifying if the permission is actually used) while providing concrete evidence of malicious activity, which is superior to mere suspicion.

---

### FINAL RECOMMENDATION

**The recommended course of action is a hybrid of Path B and Path C:**

1.  **Perform Dynamic Analysis (Path C) immediately.** Do not wait for the developer's response. Check if the app calls the `Telephony` APIs to read SMS messages during runtime.
2.  **If the dynamic analysis shows no usage of the SMS permission:** Flag the app for a "Policy Violation" regarding "Excessive Permission Requests" (Path B) and require the developer to remove the permission before approval.
3.  **If the dynamic analysis shows suspicious data exfiltration:** Immediately reject the app as "Malicious/Malware" and escalate to the security operations team for a potential developer account ban.

This approach ensures that the app is objectively tested for safety while maintaining a structured, evidence-based communication channel with the developer.
============================================================

Problem:
A privacy audit finds that an Android app collects users' Aadhaar
        numbers and precise location during onboarding, but the privacy
        notice never mentions this collection and no separate consent is
        taken, as required under India's DPDP Act, 2023. Should the audit
        team demand immediate removal of the data field, require the
        developer to redesign the consent flow before re-review, or
        escalate the finding to the app store's compliance team?

Solution:
### PATH A: Immediate Remediation (The "Hard Stop")
**Strategy:** Demand immediate removal of the data collection fields from the production build and mandate a full wipe of all previously collected Aadhaar and location data from the backend servers.

*   **Reasoning:** This approach prioritizes data minimization and immediate risk mitigation. Under the DPDP Act, collecting sensitive personal data (Aadhaar) without explicit notice and consent is a violation. By forcing removal, the organization eliminates the "ongoing harm" of unauthorized data processing.
*   **Likely Outcome:** The developer stops the violation instantly. However, this may break core app functionality if the app relies on these data points for its business model, leading to significant friction and potential loss of revenue for the developer.

### PATH B: Remediation and Compliance Redesign (The "Corrective Path")
**Strategy:** Order the developer to disable the data collection fields in the current build, then require them to submit a comprehensive redesign of the consent flow (including granular, affirmative opt-ins and updated privacy notices) for a mandatory security re-review before the feature can be re-enabled.

*   **Reasoning:** This approach balances regulatory compliance with business continuity. It acknowledges that the app may need this data for legitimate purposes, but insists that the "process" must be legal. It treats the developer as a partner in compliance rather than a target for disciplinary action.
*   **Likely Outcome:** The developer retains the ability to eventually implement the feature legally. It ensures that when the data collection resumes, it is done under the strict requirements of the DPDP Act, reducing future liability.

### PATH C: Escalation to App Store Compliance (The "Regulatory/External Path")
**Strategy:** Flag the app immediately to the App Store’s (e.g., Google Play) policy enforcement team for a violation of "User Data Policies" and "Legal Compliance," while simultaneously notifying the developer that the app is under suspension review.

*   **Reasoning:** This strategy assumes the violation is either malicious or indicative of systemic negligence. By involving the platform holder, the auditor shifts the burden of enforcement to the platform, which has the power to delist or ban the app entirely.
*   **Likely Outcome:** This carries the highest severity. The app faces potential removal from the store, which is catastrophic for the business. This is the most effective path if the audit reveals willful non-compliance or if the developer refuses to cooperate with Path A or B.

---

### EVALUATION
*   **Path A** is aggressive but lacks a long-term solution for the developer’s feature requirements.
*   **Path B** is the most balanced; it addresses the violation while providing a clear roadmap for the developer to achieve compliance. It minimizes the risk of legal liability for both the auditor and the developer.
*   **Path C** is high-risk for the developer and should be reserved for cases of bad faith. If used prematurely, it damages the relationship between the audit team and the development team and invites unnecessary scrutiny from the platform holder.

**Least Risk:** **Path B** carries the least risk. It provides a structured, documented path to compliance that satisfies the DPDP Act’s requirements for "notice and consent" while preventing the "nuclear option" of platform delisting.

---

### FINAL RECOMMENDATION
**Adopt Path B.** 

The audit team should issue a formal notice of non-compliance to the developer, requiring them to:
1.  **Cease and Desist:** Disable the data collection fields in the current version via a hotfix.
2.  **Purge/Anonymize:** Securely delete the existing, illegally collected data in accordance with DPDP requirements.
3.  **Redesign:** Submit a new onboarding flow that includes a "Notice" (detailing the purpose and necessity of Aadhaar/location data) and a "Consent" mechanism (separate, affirmative action).
4.  **Re-audit:** Provide the updated flow for a secondary security review. 

Only if the developer fails to comply with this timeline or attempts to circumvent the requirements should the team escalate to **Path C**.
============================================================

Problem:
A permissions/intents classifier flags a batch of 5,000 apps as
        suspicious with only 60% confidence. Should the team auto-quarantine
        all of them, manually review each one, or raise the confidence
        threshold and re-run the batch?

Solution:
### PATH A: Aggressive Automated Quarantine
**Reasoning:** This approach prioritizes security over availability. By auto-quarantining all 5,000 apps, the team eliminates the immediate threat of potential malware execution across the ecosystem. It assumes that a 60% confidence rating is sufficient to justify a "better safe than sorry" posture.
**Likely Outcome:** A massive surge in false positives. The team will face significant backlash from developers and users due to legitimate apps being blocked. This will create a bottleneck of support tickets and urgent manual review requests, potentially overwhelming the security team.

### PATH B: Manual Triage and Review
**Reasoning:** This approach treats the 60% confidence as an "investigative lead" rather than a definitive verdict. The team utilizes a tiered manual review process, starting with high-risk permission combinations or apps with high download counts, while allowing the rest to remain active but monitored.
**Likely Outcome:** High accuracy and minimal disruption to the ecosystem. However, this is extremely resource-intensive. If the team does not have a large dedicated workforce, the review process could take weeks, leaving potentially malicious apps active for too long.

### PATH C: Threshold Tuning and Iterative Re-analysis
**Reasoning:** This approach recognizes that the current classifier settings are too "noisy." By raising the confidence threshold (e.g., to 85% or 90%), the team filters out the low-confidence noise and isolates the most likely malicious candidates. The remaining batch is then re-processed, and the team adjusts the feature set based on the results.
**Likely Outcome:** A significantly smaller, more manageable subset of "high-confidence" suspicious apps. This allows the team to focus their manual efforts where they matter most, while reducing the false-positive rate for the bulk of the 5,000 apps.

---

### EVALUATION

| Criteria | Path A (Auto-Quarantine) | Path B (Manual Review) | Path C (Threshold Tuning) |
| :--- | :--- | :--- | :--- |
| **Risk of Malicious App Escape** | Very Low | Moderate | Low |
| **Risk of False Positives** | Extremely High | Low | Moderate |
| **Operational Efficiency** | High (Immediate) | Very Low | Moderate |
| **Developer/User Impact** | Severe | Low | Low-Moderate |

*   **Path A** is operationally reckless and creates significant reputational damage.
*   **Path B** is operationally impossible for a batch of 5,000 without extreme scaling.
*   **Path C** provides a balanced, analytical approach that optimizes limited human resources while maintaining a defensible security posture.

---

### FINAL RECOMMENDATION

**The best course of action is Path C.**

The 60% confidence level is statistically insufficient to justify mass-quarantine. The team should:
1.  **Raise the threshold:** Immediately re-run the classification with a higher threshold (e.g., 85%) to identify the "high-confidence" subset.
2.  **Prioritize:** Manually review only the high-confidence subset.
3.  **Monitor:** For the remaining apps (those between 60-85% confidence), implement "Runtime Observability" or "Enhanced Monitoring" (e.g., sandboxing or increased logging) rather than outright blocking.
4.  **Feedback Loop:** Use the findings from the manual reviews to retrain the classifier, reducing the number of future false positives.
============================================================


"""