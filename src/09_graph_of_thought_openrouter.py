"""
Title: Graph of Thought Prompting (OpenRouter)
uv run python src/09_graph_of_thought_openrouter.py
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
        temperature=0.5,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Graph-of-Thought prompt
    got_prompt = ChatPromptTemplate.from_template("""
        You are an experienced security architect.

        Solve the following security design problem by building a graph
        of interconnected reasoning nodes.

        Problem:

        {problem}

        Instructions:

        1. Identify 4-5 key sub-questions or concepts (nodes) needed to solve this problem.
        2. For each node, identify its dependencies (which other nodes it relies on).
        3. Solve the nodes in dependency order (solve dependencies first).
        4. Show how insights from one node inform others.
        5. Synthesize all nodes into a final comprehensive answer.

        Use this format for each node:
        NODE [name]: [description]
        DEPENDS ON: [list of nodes or "none"]
        SOLUTION: [reasoning and answer for this node]

        Solution:
    """)

    # Create the chain
    chain = got_prompt | llm | parser

    # Sample security design problems
    problems = [
        """
        Design a strategy for an organisation to migrate from a
        perimeter-based security model to a zero-trust architecture,
        while maintaining business continuity and staying within a
        limited security budget.
        """,

        """
        How should a company build an insider-threat detection program
        that balances employee privacy, detection accuracy, and legal
        compliance?
        """,

        """
        Design an approach for a SOC to reduce alert fatigue while
        maintaining detection coverage across cloud, endpoint, and
        network telemetry.
        """,
    ]

    # Invoke the chain for each problem, one at a time
    for i, problem in enumerate(problems, 1):
        print(f"\n{'=' * 60}")
        print(f"PROBLEM {i}")
        print("=" * 60)
        print(f"\n{problem.strip()}\n")
        print("-" * 60)

        result = chain.invoke({"problem": problem})
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

Design a strategy for an organisation to migrate from a
        perimeter-based security model to a zero-trust architecture,
        while maintaining business continuity and staying within a
        limited security budget.

------------------------------------------------------------
### Solution:

#### NODE [Current Security Model Analysis]: Analyze the current perimeter-based security model and identify its strengths and weaknesses.
DEPENDS ON: none
SOLUTION: 
To transition from a perimeter-based security model to zero-trust, it's crucial to first understand the existing security architecture. This involves identifying how data is protected within the network boundaries, what security controls are in place, and how they interact with business operations. Key aspects include:
- **Firewalls and Network Segmentation**: How these are configured and enforced.
- **Access Control Mechanisms**: Whether they are based on roles, IP addresses, or other criteria.
- **Data Protection Measures**: Encryption, data loss prevention (DLP), etc.
- **Incident Response Plan**: Current processes for detecting and responding to security incidents.

Understanding these elements will help identify areas that need reinforcement or rethinking as part of a zero-trust strategy.

---

#### NODE [Zero-Trust Architecture Components]: Identify the core components required for a zero-trust architecture.
DEPENDS ON: Current Security Model Analysis
SOLUTION: 
A zero-trust architecture focuses on several key components:
1. **Identity and Access Management (IAM)**: Ensures only authorized users have access to resources.
2. **Least Privilege Principle**: Users and devices are granted only the minimum necessary permissions.
3. **Continuous Verification**: Regularly validate user and device identities before granting access.
4. **Micro-Segmentation**: Divide the network into smaller segments to limit lateral movement.
5. **Encryption**: Encrypt data both at rest and in transit.
6. **Behavioral Analytics**: Monitor user and system behavior to detect anomalies.

These components will form the foundation of the new security model, ensuring that every connection and interaction is verified and validated.

---

#### NODE [Budget Constraints]: Determine how to allocate the limited security budget effectively.
DEPENDS ON: Zero-Trust Architecture Components
SOLUTION: 
Given the limited budget, prioritize investments in critical areas:
1. **IAM Solutions**: Implement robust identity management systems to enforce least privilege and continuous verification.
2. **Network Segmentation Tools**: Deploy tools to facilitate micro-segmentation and secure communication between segments.
3. **Behavioral Analytics Platforms**: Invest in solutions that can monitor and analyze network traffic and user behavior.
4. **Encryption Technologies**: Ensure data is encrypted both at rest and in transit using strong encryption standards.
5. **Automation Tools**: Use automation to reduce manual effort and potential human error.

Allocate funds to these areas while considering cost-effective alternatives and phased implementations to ensure gradual but steady progress towards zero-trust.

---

#### NODE [Business Continuity Plan]: Develop a plan to maintain business continuity during the transition.
DEPENDS ON: Zero-Trust Architecture Components, Budget Constraints
SOLUTION: 
To maintain business continuity, develop a comprehensive transition plan:
1. **Phased Implementation**: Gradually roll out zero-trust components to minimize disruption.
2. **Parallel Operations**: Run old and new systems in parallel during the transition phase.
3. **Communication Strategy**: Clearly communicate changes to stakeholders to manage expectations.
4. **Fallback Mechanisms**: Ensure there are backup plans in case certain components fail.
5. **Training Programs**: Educate employees about the new security measures and their role in maintaining security.

By carefully planning and executing the transition, the organization can minimize downtime and ensure smooth operation throughout the process.

---

#### NODE [Risk Assessment and Mitigation]: Conduct a risk assessment to identify potential vulnerabilities and develop mitigation strategies.
DEPENDS ON: Business Continuity Plan, Zero-Trust Architecture Components
SOLUTION: 
Conduct a thorough risk assessment to identify potential vulnerabilities and develop mitigation strategies:
1. **Identify Threats**: Assess threats specific to the organization’s industry and environment.
2. **Assess Vulnerabilities**: Evaluate existing security controls against identified threats.
3. **Develop Mitigation Strategies**: For each identified risk, develop specific mitigation strategies.
4. **Regular Audits**: Perform regular audits to ensure compliance and effectiveness of security controls.
5. **Incident Response Plan**: Enhance incident response capabilities to quickly address any breaches.

This step ensures that the organization is prepared for potential security incidents and can respond effectively to protect sensitive information and maintain business operations.

---

### Synthesis:

To migrate from a perimeter-based security model to a zero-trust architecture while maintaining business continuity and staying within a limited security budget, the organization should follow these steps:

1. **Analyze the Current Security Model**: Understand the existing security architecture to identify strengths and weaknesses.
2. **Identify Zero-Trust Components**: Determine the essential components needed for a zero-trust architecture.
3. **Allocate Budget Effectively**: Prioritize investments in critical areas such as IAM, network segmentation, behavioral analytics, and encryption.
4. **Develop a Business Continuity Plan**: Ensure smooth transition by implementing a phased approach, running parallel operations, and providing training.
5. **Conduct Risk Assessment and Mitigation**: Identify potential vulnerabilities and develop strategies to mitigate risks.

By addressing these nodes in dependency order, the organization can systematically transition to a zero-trust architecture, ensuring enhanced security without compromising business operations or exceeding the available budget.

============================================================
PROBLEM 2
============================================================

How should a company build an insider-threat detection program
        that balances employee privacy, detection accuracy, and legal
        compliance?

------------------------------------------------------------
### Solution:

#### NODE [Employee Privacy Considerations]: What measures can be taken to protect employee privacy while still detecting insider threats?
- **DEPENDS ON:** None
- **SOLUTION:** To protect employee privacy, the company should implement a comprehensive data minimization strategy. This involves collecting only the necessary data for insider threat detection and ensuring that all data is anonymized or pseudonymized where possible. Additionally, implementing strict access controls and monitoring only specific activities related to potential insider threats can help minimize the risk of privacy breaches. Regularly training employees on data protection policies and obtaining their informed consent for data collection are also crucial steps.

#### NODE [Detection Accuracy]: How can the accuracy of insider-threat detection be improved without compromising privacy?
- **DEPENDS ON:** Employee Privacy Considerations
- **SOLUTION:** Improving detection accuracy requires a multi-faceted approach. By leveraging advanced analytics and machine learning algorithms, the system can identify patterns and anomalies that may indicate insider threats. However, these algorithms must be trained on anonymized data to respect privacy constraints. Implementing behavioral analytics that track normal user behavior can help in identifying deviations that could signal malicious activity. Regularly updating the models with new data and conducting thorough validation tests are essential to ensure high accuracy.

#### NODE [Legal Compliance]: Which legal frameworks must be considered when designing an insider-threat detection program?
- **DEPENDS ON:** Employee Privacy Considerations, Detection Accuracy
- **SOLUTION:** The program must comply with various legal frameworks such as GDPR (General Data Protection Regulation), CCPA (California Consumer Privacy Act), and local data protection laws. Ensuring compliance involves conducting a thorough risk assessment to identify potential legal risks and implementing appropriate safeguards. It’s important to have clear documentation of data handling practices, obtain necessary consents from employees, and provide transparency regarding how data is used. Regular audits and compliance checks should be conducted to ensure ongoing adherence to legal requirements.

#### NODE [Balancing Privacy and Detection]: How can the company balance the need for privacy with the requirement for effective threat detection?
- **DEPENDS ON:** Employee Privacy Considerations, Detection Accuracy, Legal Compliance
- **SOLUTION:** Balancing privacy and detection involves creating a robust framework that prioritizes both aspects. The company should establish a clear policy that outlines the scope of data collection and usage. This policy should be communicated transparently to employees and stakeholders. Implementing a layered security approach that includes both technical controls (like encryption and access controls) and organizational controls (such as regular training and awareness programs) can help achieve this balance. Continuous monitoring and feedback mechanisms should be in place to adjust the program based on evolving threats and changing legal landscapes.

#### NODE [Comprehensive Insider-Threat Detection Program]: What components should a comprehensive insider-threat detection program include?
- **DEPENDS ON:** Employee Privacy Considerations, Detection Accuracy, Legal Compliance, Balancing Privacy and Detection
- **SOLUTION:** A comprehensive insider-threat detection program should include several key components:
  - **Data Collection:** Collecting relevant data while respecting privacy constraints.
  - **Analytics and Machine Learning:** Utilizing advanced analytics and machine learning to detect anomalies.
  - **Behavioral Analytics:** Tracking normal user behavior to identify deviations.
  - **Access Controls:** Implementing strict access controls to monitor and restrict sensitive activities.
  - **Regular Audits:** Conducting regular audits to ensure compliance and effectiveness.
  - **Training and Awareness:** Providing regular training and awareness programs for employees.
  - **Policy Documentation:** Maintaining clear documentation of policies and procedures.
  - **Feedback Mechanisms:** Establishing mechanisms to gather feedback and continuously improve the program.

### Synthesis:

To build an insider-threat detection program that balances employee privacy, detection accuracy, and legal compliance, the company should take a multi-faceted approach. First, it must prioritize employee privacy by implementing data minimization strategies and ensuring data is anonymized or pseudonymized. Next, it should focus on improving detection accuracy through the use of advanced analytics and machine learning, while being mindful of privacy constraints. Legal compliance must be a top priority, requiring the company to adhere to relevant legal frameworks and conduct regular audits. Balancing privacy and detection involves creating a robust framework that respects privacy while effectively detecting threats. Finally, the program should include a range of components such as data collection, analytics, access controls, regular audits, training, and policy documentation to ensure its effectiveness and compliance.

============================================================
PROBLEM 3
============================================================

Design an approach for a SOC to reduce alert fatigue while
        maintaining detection coverage across cloud, endpoint, and
        network telemetry.

------------------------------------------------------------
### Solution:

#### NODE [1]: Define Key Metrics for Alert Fatigue
**Description:** Establish metrics to quantify alert fatigue, such as the number of alerts per day, false positive rate, and alert resolution time.
**DEPENDS ON:** none
**SOLUTION:** To effectively reduce alert fatigue, we first need to define key performance indicators (KPIs) that will help us measure the current state of alert fatigue. These KPIs include:
- **Number of Alerts Per Day:** This metric helps understand the volume of alerts generated by the SOC.
- **False Positive Rate:** This measures the percentage of alerts that are incorrect or irrelevant.
- **Alert Resolution Time:** This indicates how long it takes to address each alert.

By defining these metrics, we can identify areas where alert fatigue is most prevalent and prioritize improvements.

#### NODE [2]: Identify Common Attack Patterns and Threat Indicators
**Description:** Develop a list of common attack patterns and threat indicators that the SOC should focus on.
**DEPENDS ON:** NODE [1]
**SOLUTION:** Understanding common attack patterns and threat indicators is crucial for reducing alert fatigue while maintaining detection coverage. By focusing on these patterns, the SOC can prioritize alerts that are more likely to be relevant. For example, if phishing attacks are a significant threat in the organization, alerts related to suspicious email activity should be given higher priority.

#### NODE [3]: Implement Automated Correlation Rules
**Description:** Develop automated correlation rules to filter out redundant alerts based on predefined criteria.
**DEPENDS ON:** NODE [1], NODE [2]
**SOLUTION:** Automated correlation rules can significantly reduce alert fatigue by filtering out redundant alerts. These rules can be based on the common attack patterns and threat indicators identified earlier. For instance, if multiple alerts are generated from the same IP address within a short timeframe, they can be correlated into a single alert. This reduces the number of alerts without compromising detection coverage.

#### NODE [4]: Enhance Alert Prioritization Mechanisms
**Description:** Develop a mechanism to prioritize alerts based on their severity and relevance.
**DEPENDS ON:** NODE [1], NODE [2], NODE [3]
**SOLUTION:** To ensure that critical alerts are addressed promptly, an effective prioritization mechanism is essential. This can be achieved through:
- **Severity Levels:** Assigning severity levels to alerts based on potential impact.
- **Relevance Scoring:** Using machine learning models to score alerts based on their likelihood of being relevant.
- **Contextual Information:** Incorporating contextual information such as user behavior and network anomalies to enhance alert prioritization.

By implementing these mechanisms, the SOC can focus on the most critical alerts first, reducing the overall volume of alerts without missing important threats.

#### NODE [5]: Continuous Monitoring and Feedback Loop
**Description:** Establish a continuous monitoring and feedback loop to refine alert handling processes over time.
**DEPENDS ON:** NODE [1], NODE [2], NODE [3], NODE [4]
**SOLUTION:** A continuous monitoring and feedback loop ensures that the alert handling processes are continually refined. This involves:
- **Regular Audits:** Conducting regular audits to assess the effectiveness of alert reduction strategies.
- **User Feedback:** Gathering feedback from SOC analysts to identify areas for improvement.
- **Performance Tuning:** Adjusting alert thresholds and correlation rules based on real-world performance data.

This iterative process helps maintain optimal alert coverage while minimizing alert fatigue.

### Synthesis

To solve the problem of reducing alert fatigue while maintaining detection coverage across cloud, endpoint, and network telemetry, the following comprehensive approach can be implemented:

1. **Define Key Metrics for Alert Fatigue:** Establish metrics to quantify alert fatigue, including the number of alerts per day, false positive rate, and alert resolution time.
2. **Identify Common Attack Patterns and Threat Indicators:** Develop a list of common attack patterns and threat indicators that the SOC should focus on.
3. **Implement Automated Correlation Rules:** Develop automated correlation rules to filter out redundant alerts based on predefined criteria.
4. **Enhance Alert Prioritization Mechanisms:** Develop a mechanism to prioritize alerts based on their severity and relevance.
5. **Continuous Monitoring and Feedback Loop:** Establish a continuous monitoring and feedback loop to refine alert handling processes over time.

By addressing these nodes in dependency order, the SOC can effectively reduce alert fatigue while ensuring comprehensive detection coverage across all telemetry sources.

"""