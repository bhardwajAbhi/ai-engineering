"""
Title: Self-Consistency Prompting (Google Generative AI)

Self-consistency prompting asks the model the same question several
times with some randomness, so each run reasons a little differently,
then picks the answer that appears most often (a majority vote).
This version counts the votes by hand with a dictionary and loops.

uv run python src/13_self_consistency_google.py
"""

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def extract_final_answer(response: str) -> str:
    """Find the last line that has "FINAL ANSWER:" and return the text after it."""

    # Look from the last line backwards
    for line in reversed(response.splitlines()):
        line = line.upper()
        if "FINAL ANSWER:" in line:
            answer = line.split("FINAL ANSWER:")[-1]

            # Remove bold markers and commas, then trim spaces and dots
            answer = answer.replace("*", "").replace(",", "")
            return answer.strip(" .")

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

    # Step 1: count how many times each answer appears
    vote_counts = {}
    for answer in answers:
        if answer in vote_counts:
            vote_counts[answer] += 1
        else:
            vote_counts[answer] = 1

    # Step 2: find the answer with the highest count
    # (if two answers tie, the one that appeared first wins)
    winner = None
    highest_count = 0
    for answer, count in vote_counts.items():
        if count > highest_count:
            highest_count = count
            winner = answer

    # Step 3: confidence is the share of votes the winner got
    confidence = highest_count / n_samples

    return {
        "final_answer": winner,
        "confidence": confidence,
        "vote_distribution": vote_counts,
        "samples": reasoning_samples,
    }


def main():
    load_dotenv()

    # Initialize the model
    # A temperature above 0 gives each sample a different reasoning path.
    # At 0 every sample would be almost the same and voting would add nothing.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.7,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Create the reasoning prompt
    cot_prompt = ChatPromptTemplate.from_template("""
        You are a mobile app security and data protection expert.

        Solve this problem step by step.
        Show your reasoning, then give the final answer.

        PROBLEM: {problem}

        Let's think step by step:

        [After reasoning, state your final answer on its own line as:
        FINAL ANSWER: <answer>]
    """)

    # Create the chain
    chain = cot_prompt | llm | parser

    # A tricky problem: people aged exactly 18 are adults, so not counted
    problem = """
        Under India's DPDP Act, 2023, a child is a person who has not
        completed 18 years of age, and a data fiduciary needs verifiable
        parental consent to process a child's personal data. An app has
        5,000 users: 900 are under 13, 1,100 are between 13 and 17, 600
        are 18 years old and 2,400 are 19 or older. How many users need
        verifiable parental consent?
        Answer with a number only.
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

Under India's DPDP Act, 2023, a child is a person who has not
        completed 18 years of age, and a data fiduciary needs verifiable
        parental consent to process a child's personal data. An app has
        5,000 users: 900 are under 13, 1,100 are between 13 and 17, 600
        are 18 years old and 2,400 are 19 or older. How many users need
        verifiable parental consent?
        Answer with a number only.

============================================================
Generating multiple reasoning paths...

Sample 1: 2000
Sample 2: 2000
Sample 3: 2000
Sample 4: 2000
Sample 5: 2000

============================================================
FINAL ANSWER: 2000
CONFIDENCE: 100%
VOTE DISTRIBUTION: {'2000': 5}

"""