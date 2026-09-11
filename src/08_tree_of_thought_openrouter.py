"""
Title: Tree of Thought Prompting (OpenRouter)
uv run python src/08_tree_of_thought_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main():
    load_dotenv()

    # Initialize the model
    # A higher temperature encourages the three paths to genuinely differ
    # instead of converging on the same reasoning.
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Tree-of-Thought prompt
    tot_prompt = ChatPromptTemplate.from_template("""
        You are an experienced incident response strategist.

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
        Ransomware has encrypted a hospital's patient records system.
        Backups exist but are 18 hours old. Should the organisation pay
        the ransom, restore from backups, or attempt in-house decryption?
        """,

        """
        A SOC has a backlog of 400 unreviewed alerts after a public
        holiday. Should the team triage by severity, by asset criticality,
        or strictly in chronological order?
        """,

        """
        A zero-day exploit is disclosed for software running on 60 percent 
        of the company's servers. No official patch exists yet. Should the
        team isolate the affected servers now, apply an unofficial
        community mitigation, or wait for the vendor patch?
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
Ransomware has encrypted a hospital's patient records system.
        Backups exist but are 18 hours old. Should the organisation pay
        the ransom, restore from backups, or attempt in-house decryption?

Solution:
### PATH A: Restore from Backups

**Reasoning:**
Restoring from backups is a straightforward and potentially quick way to recover the patient records system. Since the backups are only 18 hours old, they likely contain all critical data that has been updated since then. This method minimizes downtime and ensures that the system is restored to a known good state.

**Likely Outcome:**
- **Success:** The system will be restored within a few hours.
- **Failure:** If the ransomware has modified the backup files or if there are issues with the restoration process, the system might not fully recover.

### PATH B: Attempt In-House Decryption

**Reasoning:**
Given that the backups are only 18 hours old, attempting to decrypt the current encrypted files in-house could provide an alternative to paying the ransom. This approach requires specialized knowledge and tools but could potentially save the organization significant money and avoid reputational damage associated with paying ransoms.

**Likely Outcome:**
- **Success:** If the decryption tools and methods are available and effective, the system can be decrypted without paying.
- **Failure:** If the decryption fails, the organization may still need to pay the ransom or use the backups, which might be outdated.

### PATH C: Pay the Ransom

**Reasoning:**
Paying the ransom is a direct and guaranteed way to regain access to the encrypted data. However, this option comes with significant risks, including the possibility of the attackers not delivering the decryption key, additional demands, or future attacks by the same group.

**Likely Outcome:**
- **Success:** The organization regains access to the encrypted data.
- **Failure:** The attackers do not provide the decryption key, leading to further loss of data or financial loss due to the ransom payment.

### EVALUATION:

- **PATH A (Restore from Backups):** This path carries moderate risk as it relies on the integrity of the backup process and the availability of up-to-date data. However, it is a tried-and-true method that minimizes downtime and avoids the ethical and financial implications of paying a ransom.
  
- **PATH B (Attempt In-House Decryption):** This path carries higher technical risk but offers the potential for long-term savings and avoiding reputational damage. It also allows the organization to retain control over its data and processes.

- **PATH C (Pay the Ransom):** This path carries the highest risk, including the possibility of not receiving the decryption key, potential future attacks, and the ethical implications of supporting criminal activity.

### FINAL RECOMMENDATION:

Based on the analysis, **PATH A (Restore from Backups)** is the best course of action. While it does carry some risk, it is the most balanced approach. Restoring from backups minimizes downtime, ensures data integrity, and avoids the ethical and financial pitfalls associated with paying a ransom. Additionally, the organization should consider enhancing its incident response plan to include regular and robust encryption practices, continuous monitoring, and rapid backup procedures to mitigate similar incidents in the future.
============================================================

Problem:
A SOC has a backlog of 400 unreviewed alerts after a public
        holiday. Should the team triage by severity, by asset criticality,
        or strictly in chronological order?

Solution:
### Solution:

#### PATH A:
**Strategy: Triage by Severity**
- **Reasoning:** Prioritizing alerts based on severity ensures that the most critical issues are addressed first. Alerts marked as high severity typically indicate a significant risk or potential compromise, such as unauthorized access attempts, malware infections, or data exfiltration.
- **Outcome:** The SOC team will focus on addressing the most pressing threats, potentially mitigating severe security incidents before they escalate. This approach minimizes the risk of critical vulnerabilities being overlooked and ensures that the most impactful alerts are handled promptly.

#### PATH B:
**Strategy: Triage by Asset Criticality**
- **Reasoning:** Triage by asset criticality involves prioritizing alerts based on the importance of the affected assets. Critical assets, such as those handling sensitive data or supporting core business functions, would be given higher priority.
- **Outcome:** This approach ensures that any alert affecting mission-critical systems is reviewed and acted upon quickly. It helps in protecting the organization's most valuable resources and reducing the potential impact of security breaches.

#### PATH C:
**Strategy: Triage in Chronological Order**
- **Reasoning:** Reviewing alerts in chronological order means the team starts with the oldest alerts and works their way through to the newest. This method assumes that newer alerts are more likely to be relevant and actionable.
- **Outcome:** This strategy may not prioritize the most urgent or critical alerts but can help in maintaining a consistent workflow and ensuring that no alerts are missed. However, it risks overlooking time-sensitive issues that could have escalated since the alerts were generated.

### EVALUATION:

1. **PATH A (Triage by Severity):**
   - **Risk:** Lower risk of missing critical alerts because severe alerts are prioritized first.
   - **Efficiency:** Ensures the most impactful issues are addressed quickly.
   - **Flexibility:** Can adapt to changing circumstances if new high-severity alerts come in.

2. **PATH B (Triage by Asset Criticality):**
   - **Risk:** Lower risk of overlooking critical assets, ensuring essential systems remain protected.
   - **Efficiency:** Streamlines the process by focusing on the most important parts of the infrastructure.
   - **Flexibility:** Allows for dynamic prioritization based on the asset's role within the organization.

3. **PATH C (Triage in Chronological Order):**
   - **Risk:** Higher risk of missing critical alerts that may have become more urgent since they were initially generated.
   - **Efficiency:** Maintains a structured and consistent process.
   - **Flexibility:** Less adaptable to immediate changes in threat landscape.

### FINAL RECOMMENDATION:

**Recommended Course of Action: PATH A (Triage by Severity)**

**Reasoning:** While both PATH B and PATH C have merits, triaging by severity offers the most robust protection against critical security incidents. By focusing on the most severe alerts first, the SOC team ensures that the highest-priority threats are addressed promptly, thereby minimizing the risk of significant damage. This approach aligns with best practices in incident response and provides the necessary flexibility to adapt to evolving threats.
============================================================

Problem:
A zero-day exploit is disclosed for software running on 60 percent 
        of the company's servers. No official patch exists yet. Should the
        team isolate the affected servers now, apply an unofficial
        community mitigation, or wait for the vendor patch?

Solution:
### Solution:

#### PATH A:
**Strategy:** Isolate the affected servers now.

**Reasoning:**
Isolating the affected servers immediately is a proactive approach to mitigate potential damage. This strategy minimizes the risk of the exploit being exploited by malicious actors, as it prevents the affected systems from communicating with other parts of the network. Additionally, isolating the servers can help contain any lateral movement if an attacker gains access to one of the isolated systems.

**Likely Outcome:**
- **Immediate Mitigation:** The risk of the exploit being used to spread further within the network is significantly reduced.
- **Operational Impact:** There may be a temporary disruption in services running on the isolated servers, which could affect business operations.
- **Resource Utilization:** Additional resources will be needed to manage and monitor the isolated servers.
- **Risk:** If the exploit is highly sophisticated, it might still find a way to bypass isolation measures, leading to potential compromise.

#### PATH B:
**Strategy:** Apply an unofficial community mitigation.

**Reasoning:**
Using an unofficial community mitigation can provide a quick fix while waiting for an official patch from the vendor. Community-developed patches often emerge faster than official ones and can address the immediate security concern. However, these patches are not officially vetted and may have unforeseen issues or vulnerabilities.

**Likely Outcome:**
- **Quick Fix:** The system is protected against the exploit more quickly than waiting for an official patch.
- **Uncertainty:** There is a risk that the community mitigation may introduce new vulnerabilities or bugs, potentially leading to system instability.
- **Security Risk:** The unofficial nature of the patch means it has not been thoroughly tested, which could lead to security issues if it is not fully compatible with the existing software.
- **Risk:** The community mitigation might not fully address the exploit, leaving the system vulnerable.

#### PATH C:
**Strategy:** Wait for the vendor patch.

**Reasoning:**
Waiting for the vendor patch is a conservative approach. Official patches are typically well-tested and ensure compatibility with the software version. Waiting also allows the vendor to address any potential issues and provide a comprehensive solution.

**Likely Outcome:**
- **Comprehensive Solution:** Once the official patch is released, it will likely address the exploit comprehensively and include all necessary fixes.
- **Delayed Response:** The system remains vulnerable until the patch is available, which could be several days or even weeks.
- **Risk:** During this period, the system is exposed to the exploit, and there is no guarantee that other exploits or vulnerabilities may not be discovered.
- **Resource Utilization:** No additional resources are required beyond monitoring the situation.

### EVALUATION:

- **PATH A (Isolation):** Provides immediate protection but may cause operational disruptions and requires additional resource management.
- **PATH B (Community Mitigation):** Offers a quick fix but comes with the risk of introducing new vulnerabilities and potential system instability.
- **PATH C (Vendor Patch):** Ensures a comprehensive solution but leaves the system exposed during the wait period.

### FINAL RECOMMENDATION:

**Recommended Course of Action:**

Given the need for immediate protection while ensuring long-term stability, **Path A (Isolation)** appears to be the best course of action. While it introduces some operational disruption, it provides an immediate barrier against the exploit, minimizing the risk of further damage. Once the official patch becomes available, the isolation can be lifted, and the system can be updated without the risk of the exploit being exploited again.

**Action Plan:**
1. Immediately isolate the affected servers.
2. Monitor the network for any signs of compromise.
3. Once the official patch is available, apply it and lift the isolation measures.
4. Conduct a thorough post-patch assessment to ensure full recovery and security.

This approach balances immediate security needs with long-term stability and reduces overall risk.
============================================================


"""    