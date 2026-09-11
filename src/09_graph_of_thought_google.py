"""
Title: Graph of Thought Prompting (Google Generative AI)
uv run python src/09_graph_of_thought_google.py
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
        temperature=0.5,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the Graph-of-Thought prompt
    got_prompt = ChatPromptTemplate.from_template("""
        You are an Android application security engineer.

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
        Design a strategy for an Android malware detection pipeline that
        balances detection accuracy, a low false-positive rate, and fast
        inference time for on-device scanning.
        """,

        """
        How should an app-store review team design a permission-risk
        scoring system that accounts for app category, requested
        permissions, and developer reputation?
        """,

        """
        Design an approach for continuously retraining a permissions and
        intents malware classifier as new malware families emerge, while
        avoiding catastrophic forgetting of older threats.
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

Design a strategy for an Android malware detection pipeline that
        balances detection accuracy, a low false-positive rate, and fast
        inference time for on-device scanning.

------------------------------------------------------------
Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.

To design an efficient, high-accuracy Android malware detection pipeline, we must balance computational constraints (on-device) with the complexity of modern obfuscation techniques.

### Graph of Interconnected Reasoning Nodes

#### NODE 1: Feature Extraction Strategy
**DEPENDS ON:** None
**SOLUTION:** To balance speed and accuracy, we must avoid heavy dynamic analysis (which drains battery/CPU). We will use a **Hybrid Feature Set**:
*   **Static Metadata:** Manifest permissions, API call signatures, and intent filters (Low cost).
*   **Structural Features:** Control Flow Graph (CFG) hashes and opcode frequency distributions (Medium cost).
*   **Lightweight Heuristics:** Detection of known malicious code patterns (e.g., dynamic code loading, reflection usage) via static analysis.
*   *Insight:* By focusing on structural metadata rather than full de-obfuscation, we maintain high speed while capturing the "DNA" of the application.

#### NODE 2: Inference Model Architecture
**DEPENDS ON:** Node 1
**SOLUTION:** We will utilize a **Two-Tiered Detection Architecture**:
*   **Tier 1 (The Filter):** A lightweight Random Forest or shallow Neural Network running on-device. It processes simple metadata from Node 1 to perform "fast-path" classification.
*   **Tier 2 (The Deep Inspector):** If Tier 1 returns a "Suspicious" score, the app triggers a more complex Transformer-based model (trained on opcode sequences) or offloads the analysis to a secure backend cloud service.
*   *Insight:* This tiered approach ensures >95% of benign apps are cleared instantly, while complex malware is handled by more robust (but slower) inference.

#### NODE 3: False-Positive Mitigation & Context
**DEPENDS ON:** Node 1, Node 2
**SOLUTION:** High false-positive rates are often caused by legitimate apps using aggressive obfuscation (e.g., banking apps). We will implement:
*   **Allow-listing (Reputation):** Integrate with Google Play Protect or a trusted developer hash database.
*   **Behavioral Context:** Instead of just static analysis, we monitor *runtime* behavior (e.g., does the app attempt to access contacts while in the background?).
*   *Insight:* By combining static features with runtime context, we reduce false positives triggered by apps that "look" malicious but behave legitimately.

#### NODE 4: Resource-Aware Orchestration
**DEPENDS ON:** Node 2, Node 3
**SOLUTION:** To ensure the pipeline doesn't degrade user experience, we implement an **Orchestration Layer**:
*   **Trigger-based Scanning:** Only scan on install, update, or when high-risk permissions are requested.
*   **Thermal/Battery Throttling:** If the device is low on battery or overheating, the "Deep Inspector" (Node 2) is deferred or offloaded to the cloud.
*   *Insight:* The detection pipeline must be "invisible" to the user, prioritizing system stability over immediate scanning.

---

### Synthesis: Comprehensive Strategy

The proposed Android malware detection pipeline is a **Context-Aware Tiered Pipeline**:

1.  **Efficiency via Tiering:** We prioritize speed by using a **Tier 1 lightweight classifier** that analyzes static metadata and structural features (Node 1/2). This allows the system to process the vast majority of apps in milliseconds.
2.  **Accuracy via Deep Inspection:** Only apps flagged by the Tier 1 filter are subjected to the **Tier 2 Deep Inspector**. This model uses Transformer-based analysis on opcode sequences to identify sophisticated malware that static analysis misses.
3.  **False-Positive Reduction:** We integrate a **Reputation-based Allow-list** and **Behavioral Context** (Node 3). By checking if an app has a verified developer signature or if its runtime behavior matches its static profile, we significantly lower the false-positive rate.
4.  **User-Centric Orchestration:** The entire process is managed by an **Orchestration Layer** (Node 4) that defers intensive tasks to the cloud or low-activity periods, ensuring the security pipeline does not impact battery life or device performance.

**Final Outcome:** This design achieves a low-latency "fast-path" for benign apps, a high-accuracy "deep-path" for threats, and a resource-conscious management system that minimizes the performance tax on the user's device.

============================================================
PROBLEM 2
============================================================

How should an app-store review team design a permission-risk
        scoring system that accounts for app category, requested
        permissions, and developer reputation?

------------------------------------------------------------
As an Android application security engineer, I have structured the design of a permission-risk scoring system into a dependency-based graph.

### The Graph Nodes

1.  **NODE [Permission Contextualization]:** Defining the "Normal" vs. "Dangerous" baseline per category.
2.  **NODE [Developer Trust Scoring]:** Establishing a reputation metric based on historical behavior.
3.  **NODE [Risk Scoring Algorithm]:** Integrating permissions and developer history into a unified score.
4.  **NODE [Feedback Loop & Thresholding]:** Implementing dynamic adjustments based on real-world incident data.

---

### Node Solutions

**NODE [Permission Contextualization]**
*   **DEPENDS ON:** None
*   **SOLUTION:** Permissions cannot be evaluated in a vacuum. We must map permissions to app categories (e.g., a "Flashlight" app requesting `ACCESS_FINE_LOCATION` is an anomaly, whereas a "Navigation" app requesting it is expected). We categorize permissions into risk tiers (Runtime vs. Install-time) and define "Category-Permission Affinity" matrices. If a permission is outside the "expected" scope of a category, it triggers a high-risk flag regardless of other factors.

**NODE [Developer Trust Scoring]**
*   **DEPENDS ON:** None
*   **SOLUTION:** We create a reputation score based on: (a) Age of the developer account, (b) History of policy violations/malware removals, (c) App update frequency, and (d) Code signing consistency. A high-trust developer (long history, zero violations) acts as a "risk buffer," allowing for slightly more complex permission requests, while new or flagged developers face stricter scrutiny.

**NODE [Risk Scoring Algorithm]**
*   **DEPENDS ON:** Permission Contextualization, Developer Trust Scoring
*   **SOLUTION:** We use a weighted formula: `Risk Score = (Permission_Deviation_Weight * Permission_Risk) / Developer_Trust_Score`.
    *   `Permission_Deviation_Weight`: How far the requested permission deviates from the category norm.
    *   `Permission_Risk`: The inherent sensitivity of the requested API (e.g., `READ_SMS` > `CAMERA`).
    *   `Developer_Trust_Score`: A multiplier that decreases the total risk if the developer is highly reputable.

**NODE [Feedback Loop & Thresholding]**
*   **DEPENDS ON:** Risk Scoring Algorithm
*   **SOLUTION:** The system must be dynamic. If an app with a "Medium" risk score is later found to be exfiltrating data, the system automatically retroactively lowers the reputation of the developer and increases the `Permission_Deviation_Weight` for that specific permission in that category. This ensures the model learns from emerging threat patterns (e.g., new malware delivery techniques).

---

### Synthesis: Comprehensive Design

To build an effective permission-risk scoring system, the review team should implement a **Multi-Factor Risk Engine**:

1.  **Baseline Categorization:** Start by mapping every app category to a set of "Expected Permissions." This creates a baseline of "normalcy." Any deviation (e.g., a Calculator app asking for `READ_CONTACTS`) is automatically assigned a high-risk penalty.
2.  **Reputation-Adjusted Scoring:** Use the Developer Trust Score as a "Risk Multiplier." A developer with a clean, long-standing record earns a "Trust Discount," allowing them to request sensitive permissions with less friction. Conversely, new developers or those with prior policy strikes are subjected to a "High-Scrutiny Multiplier," where even standard permissions undergo manual code analysis.
3.  **The Weighted Formula:** The final score is not binary. It is a gradient. Apps scoring below a certain threshold are auto-approved; those in the middle are sent for automated static/dynamic analysis (SAST/DAST); those above the threshold are flagged for manual human review.
4.  **Continuous Learning:** The system must integrate with the Play Store’s post-install telemetry. If a specific developer or category shows a spike in malicious activity, the `Risk Scoring Algorithm` should automatically tighten the thresholds for those specific parameters.

**Final Result:** This design moves the review team from a static, manual process to a **risk-based, automated triage system** that prioritizes human attention on the most suspicious applications while streamlining the submission process for high-trust, low-risk developers.

============================================================
PROBLEM 3
============================================================

Design an approach for continuously retraining a permissions and
        intents malware classifier as new malware families emerge, while
        avoiding catastrophic forgetting of older threats.

------------------------------------------------------------
To design a robust, continuously evolving malware classifier for Android (permissions and intents), we must balance the acquisition of new threat intelligence with the preservation of historical detection capabilities.

### Graph of Reasoning Nodes

#### NODE 1: Data Drift & Concept Evolution
**DESCRIPTION:** Defining how to detect when the feature distribution (permissions/intents) of new malware significantly deviates from the training set.
**DEPENDS ON:** None
**SOLUTION:** We implement a **distributional monitoring layer** using Kullback-Leibler (KL) divergence to track shifts in the feature space. When new malware families emerge, the model encounters "out-of-distribution" samples. By quantifying this drift, we trigger the retraining pipeline only when statistically significant changes occur, preventing unnecessary compute costs and model instability.

#### NODE 2: Replay Buffer Strategy
**DESCRIPTION:** Selecting a representative subset of historical malware data to prevent catastrophic forgetting.
**DEPENDS ON:** NODE 1
**SOLUTION:** We utilize a **Core-Set selection algorithm**. Instead of storing all historical data (which is storage-prohibitive), we maintain a fixed-size "Replay Buffer." This buffer contains high-entropy samples that represent the decision boundaries of older malware families. As new data arrives (triggered by NODE 1), we use a reservoir sampling technique to ensure the buffer maintains a balanced representation of both legacy threats and the newly identified families.

#### NODE 3: Elastic Weight Consolidation (EWC)
**DESCRIPTION:** A regularization technique to protect critical weights learned from past threats.
**DEPENDS ON:** NODE 2
**SOLUTION:** To further mitigate forgetting, we apply **EWC** during the retraining process. We calculate the Fisher Information Matrix to identify which model weights are most critical for classifying historical malware. During the training on new data, we add a penalty term to the loss function that constrains updates to these "critical" weights. This allows the model to learn new patterns while keeping the "memory" of old threats intact.

#### NODE 4: Continuous Evaluation & Feedback Loop
**DESCRIPTION:** Establishing a verification mechanism to ensure the updated model maintains performance on both legacy and new threats.
**DEPENDS ON:** NODE 3
**SOLUTION:** We implement a **"Champion-Challenger" validation framework**. The current production model (Champion) and the newly retrained model (Challenger) are both evaluated against a "Golden Dataset" (a static, curated set of diverse historical and recent threats). The Challenger is only promoted to production if it meets or exceeds the Champion's F1-score on the Golden Dataset, ensuring no regression in detection capabilities.

---

### Synthesis: Comprehensive Solution

To solve the problem of continuous retraining without catastrophic forgetting, we integrate these nodes into a **Feedback-Driven Incremental Learning Pipeline**:

1.  **Triggering:** The system monitors incoming Android application telemetry. When the **Data Drift** detector (Node 1) signals that new malware families (e.g., novel intent-based obfuscation) have emerged, the retraining pipeline is initiated.
2.  **Data Preparation:** We combine the new malware samples with a representative **Replay Buffer** (Node 2) containing historical threat vectors. This ensures the model is exposed to both the old and the new simultaneously.
3.  **Constrained Optimization:** During training, we use **EWC** (Node 3) to freeze or penalize changes to the neural network weights that were essential for identifying legacy malware. This prevents the model from "overwriting" its knowledge of older threats to accommodate the new ones.
4.  **Validation & Promotion:** Before deployment, the model undergoes a **Champion-Challenger evaluation** (Node 4). By testing against a static Golden Dataset, we guarantee that the update improves detection of new families without degrading performance on historical threats.

This architecture creates a self-sustaining security cycle: it autonomously detects when it is becoming obsolete, uses memory-efficient data management to retain past knowledge, employs weight-level regularization to prevent forgetting, and enforces rigorous validation to maintain high precision in a shifting Android threat landscape.


"""