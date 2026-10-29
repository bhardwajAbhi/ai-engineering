"""
Title: Self-Consistency Prompting (OpenRouter)

Self-consistency prompting asks the model the same question several
times with some randomness, so each run reasons a little differently,
then picks the answer that appears most often (a majority vote).
This version uses a regex to find the answer and Counter to count votes.

Note: voting does not need a number. Any short answer of a fixed form
works, such as a label or a yes or no. This version votes on a label.
Long written answers do not work, because no two runs word them alike.

uv run python src/13_self_consistency_openrouter.py
"""

import re
from collections import Counter

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_final_answer(response: str) -> str:
    """Extract the final answer (a label) from the response."""

    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
    if match:
        # Remove bold markers and dots, and ignore letter case
        # so that equal labels count as the same vote
        return match.group(1).strip("*. ").upper()

    # No "FINAL ANSWER:" line was found
    return "UNKNOWN"


def self_consistency(chain, problem: str, n_samples: int = 5) -> dict:
    """Ask the same problem several times and return the most common answer."""

    answers = []
    reasoning_samples = []

    for i in range(n_samples):
        response = chain.invoke({"problem": problem})
        answer = extract_final_answer(response)

        answers.append(answer)
        reasoning_samples.append(response)
        print(f"Sample {i + 1}: {answer}")

    # Count the votes and take the answer with the most votes
    vote_counts = Counter(answers)
    winner, count = vote_counts.most_common(1)[0]
    confidence = count / n_samples

    return {
        "final_answer": winner,
        "confidence": confidence,
        "vote_distribution": dict(vote_counts),
        "samples": reasoning_samples,
    }


def main():
    load_dotenv()

    # Initialize the model
    # A temperature above 0 gives each sample a different reasoning path.
    # At 0 every sample would be almost the same and voting would add nothing.
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0.7,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the reasoning prompt
    cot_prompt = ChatPromptTemplate.from_template("""
        You are a mobile app security expert.

        Solve this problem step by step.
        Show your reasoning, then give the final answer.

        PROBLEM: {problem}

        Let's think step by step:

        [After reasoning, state your final answer on its own line as:
        FINAL ANSWER: <answer>]
    """)

    # Create the chain
    chain = cot_prompt | llm | parser

    # A tricky labeling problem: permissions alone are not proof of harm,
    # but permissions that do not match the app's features are suspicious
    problem = """
        An app requests the READ_SMS and SEND_SMS permissions but has no
        messaging feature. No harmful behavior has been confirmed.
        Rules: label an app MALICIOUS only when harmful behavior is
        confirmed. Label it SUSPICIOUS when its permissions do not match
        its features. Otherwise label it BENIGN.
        Answer with exactly one of: BENIGN, SUSPICIOUS or MALICIOUS.
    """

    print("=" * 60)
    print("PROBLEM")
    print("=" * 60)
    print(f"\n{problem.strip()}\n")

    print("=" * 60)
    print("Generating multiple reasoning paths...\n")

    result = self_consistency(chain, problem, n_samples=5)

    print("\n" + "=" * 60)
    print(f"FINAL ANSWER: {result['final_answer']}")
    print(f"CONFIDENCE: {result['confidence']:.0%}")
    print(f"VOTE DISTRIBUTION: {result['vote_distribution']}")


if __name__ == "__main__":
    main()


"""
-----------------------------------------
Output:
-----------------------------------------


============================================================
PROBLEM
============================================================

An app requests the READ_SMS and SEND_SMS permissions but has no
        messaging feature. No harmful behavior has been confirmed.
        Rules: label an app MALICIOUS only when harmful behavior is
        confirmed. Label it SUSPICIOUS when its permissions do not match
        its features. Otherwise label it BENIGN.
        Answer with exactly one of: BENIGN, SUSPICIOUS or MALICIOUS.

============================================================
Generating multiple reasoning paths...

Sample 1: SUSPICIOUS
Sample 2: SUSPICIOUS
Sample 3: SUSPICIOUS
Sample 4: SUSPICIOUS
Sample 5: SUSPICIOUS

============================================================
FINAL ANSWER: SUSPICIOUS
CONFIDENCE: 100%
VOTE DISTRIBUTION: {'SUSPICIOUS': 5}


"""