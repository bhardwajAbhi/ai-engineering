"""
Title: RunnableLambda for Custom Processing (OpenRouter)

RunnableLambda turns an ordinary Python function into a step of the chain,
so your own code can run at any point. Here it is used before the model to
prepare the input and after the model to turn its answer into a clear
verdict with advice.

uv run python src/23_lambda_openrouter.py
"""

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.globals import set_debug


def main():
    load_dotenv()
    set_debug(True)

    # Initialize the model
    llm = ChatOpenRouter(
        model="qwen/qwen-2.5-7b-instruct",
        temperature=0,
    )

    # Parser to extract the text from the model response
    parser = StrOutputParser()

    # Custom preprocessing function
    def preprocess(text: str) -> dict:
        """Count the words and links in the message."""
        # Example:
        #   input:  "URGENT: Open http://bit.ly/kyc-upd8 now"
        #   output: {"text": "URGENT: Open http://bit.ly/kyc-upd8 now", "word_count": 4, "link_count": 1}
        cleaned = text.strip()  # remove spaces at the start and end
        words = cleaned.split()  # split the message into words
        word_count = len(words)

        link_count = 0
        for word in words:
            if word.lower().startswith(("http://", "https://")):  # a link starts with http:// or https://
                link_count += 1

        return {"text": cleaned, "word_count": word_count, "link_count": link_count}

    # Custom postprocessing function
    def format_output(result: str) -> str:
        """Turn the model's answer into a verdict, a reason and advice."""
        # Example:
        #   input:  "PHISHING: It creates panic and asks you to open a short link."
        #   output: Verdict: PHISHING
        #           Reason: It creates panic and asks you to open a short link.
        #           Advice: Do not click any link. Delete the message and report it
        #                   at cybercrime.gov.in or call 1930.
        verdict, _, reason = result.partition(":")  # split at the first colon
        verdict = verdict.strip().upper()

        if verdict == "PHISHING":
            advice = "Do not click any link. Delete the message and report it at cybercrime.gov.in or call 1930."
        elif verdict == "SAFE":
            advice = "No action needed."
        else:
            verdict = "UNCLEAR"
            reason = result
            advice = "Check the sender before you act on this message."

        return f"Verdict: {verdict}\nReason: {reason.strip()}\nAdvice: {advice}"

    # Build chain with custom functions
    prompt = ChatPromptTemplate.from_template(
        "Analyze this message (words: {word_count}, links: {link_count}) "
        "and decide whether it is phishing. Start your answer with PHISHING or SAFE, "
        "then a colon, then one short reason.\n\n"
        "Message: {text}"
    )

    # The chain runs in this order (* marks our own functions):
    #   1. preprocess*    - count the words and links
    #   2. prompt         - build the prompt
    #   3. llm and parser - the model replies, as plain text
    #   4. format_output* - turn the answer into a verdict, a reason and advice
    chain = RunnableLambda(preprocess) | prompt | llm | parser | RunnableLambda(format_output)

    print("=" * 60)
    print("CUSTOM PROCESSING WITH LAMBDAS")
    print("=" * 60)

    result = chain.invoke(
        "  URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now "
        "or your account will be BLOCKED today.  "
    )
    print(result)


if __name__ == "__main__":
    main()

"""
-----------------------------------------
Output: WITH DEBUGGING OFF
-----------------------------------------

============================================================
CUSTOM PROCESSING WITH LAMBDAS
============================================================
Verdict: PHISHING
Reason: Uses a suspicious shortened URL and threats of account blocking to create urgency.
Advice: Do not click any link. Delete the message and report it at cybercrime.gov.in or call 1930.

"""






"""
-----------------------------------------
Output: WITH DEBUGGING ON
-----------------------------------------

============================================================
CUSTOM PROCESSING WITH LAMBDAS
============================================================
[chain/start] [chain:RunnableSequence] Entering Chain run with input:
{
  "input": "  URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today.  "
}
[chain/start] [chain:RunnableSequence > chain:preprocess] Entering Chain run with input:
{
  "input": "  URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today.  "
}
[chain/end] [chain:RunnableSequence > chain:preprocess] [0ms] Exiting Chain run with output:
{
  "text": "URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today.",
  "word_count": 15,
  "link_count": 1
}
[chain/start] [chain:RunnableSequence > prompt:ChatPromptTemplate] Entering Prompt run with input:
{
  "text": "URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today.",
  "word_count": 15,
  "link_count": 1
}
[chain/end] [chain:RunnableSequence > prompt:ChatPromptTemplate] [1ms] Exiting Prompt run with output:
[outputs]
[llm/start] [chain:RunnableSequence > llm:ChatOpenRouter] Entering LLM run with input:
{
  "prompts": [
    "Human: Analyze this message (words: 15, links: 1) and decide whether it is phishing. Start your answer with PHISHING or SAFE, then a colon, then one short reason.\n\nMessage: URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today."
  ]
}
[llm/end] [chain:RunnableSequence > llm:ChatOpenRouter] [3.56s] Exiting LLM run with output:
{
  "generations": [
    [
      {
        "text": "PHISHING: It uses urgency and a suspicious shortened link to trick the user into clicking.",
        "generation_info": {
          "finish_reason": "stop",
          "logprobs": null
        },
        "type": "ChatGeneration",
        "message": {
          "lc": 1,
          "type": "constructor",
          "id": [
            "langchain",
            "schema",
            "messages",
            "AIMessage"
          ],
          "kwargs": {
            "content": "PHISHING: It uses urgency and a suspicious shortened link to trick the user into clicking.",
            "additional_kwargs": {
              "reasoning_content": "1.  **Analyze the Request:**\n    *   Input message: \"URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today.\"\n    *   Word count: 15 (matches prompt).\n    *   Link count: 1 (matches prompt).\n    *   Task: Decide if it's phishing or safe.\n    *   Format requirement: Start with \"PHISHING\" or \"SAFE\", followed by a colon, then one short reason.\n\n2.  **Analyze the Message:**\n    *   \"URGENT\": Creates a sense of urgency, a classic phishing tactic.\n    *   \"Your KYC is pending\": References a common financial/account verification process to seem legitimate.\n    *   \"Open http://bit.ly/kyc-upd8 now\": Contains a shortened URL (bit.ly), which is often used to hide malicious links. The domain name \"kyc-upd8\" mimics a legitimate KYC update process.\n    *   \"or your account will be BLOCKED today\": Another urgency/scarcity tactic to pressure the user into clicking without thinking.\n\n3.  **Determine Classification:**\n    *   The message exhibits multiple hallmarks of phishing: urgency, fear of account suspension, request to click a suspicious/shortened link, and vague but official-sounding terminology (KYC).\n    *   Classification: PHISHING.\n\n4.  **Formulate the Reason:**\n    *   Needs to be one short reason.\n    *   Draft 1: It uses urgency and a suspicious shortened link to trick the user into clicking.\n    *   Draft 2: It creates false urgency and uses a suspicious shortened link to steal credentials.\n    *   Draft 3 (shorter): It uses urgency and a suspicious shortened link to trick the user.\n\n5.  **Format the Output:**\n    *   PHISHING: It uses urgency and a suspicious shortened link to trick the user.\n\n6.  **Review against constraints:**\n    *   Starts with PHISHING or SAFE? Yes (PHISHING).\n    *   Followed by a colon? Yes.\n    *   One short reason? Yes.\n    *   Word count of message was 15, links 1 (just confirming the input matches, not a constraint on my output).\n\n7.  **Final Polish:**\n    PHISHING: It uses urgency and a suspicious shortened link to trick the user into clicking.",
              "reasoning_details": [
                {
                  "type": "reasoning.text",
                  "format": "unknown",
                  "index": 0,
                  "text": "1.  **Analyze the Request:**\n    *   Input message: \"URGENT: Your KYC is pending. Open http://bit.ly/kyc-upd8 now or your account will be BLOCKED today.\"\n    *   Word count: 15 (matches prompt).\n    *   Link count: 1 (matches prompt).\n    *   Task: Decide if it's phishing or safe.\n    *   Format requirement: Start with \"PHISHING\" or \"SAFE\", followed by a colon, then one short reason.\n\n2.  **Analyze the Message:**\n    *   \"URGENT\": Creates a sense of urgency, a classic phishing tactic.\n    *   \"Your KYC is pending\": References a common financial/account verification process to seem legitimate.\n    *   \"Open http://bit.ly/kyc-upd8 now\": Contains a shortened URL (bit.ly), which is often used to hide malicious links. The domain name \"kyc-upd8\" mimics a legitimate KYC update process.\n    *   \"or your account will be BLOCKED today\": Another urgency/scarcity tactic to pressure the user into clicking without thinking.\n\n3.  **Determine Classification:**\n    *   The message exhibits multiple hallmarks of phishing: urgency, fear of account suspension, request to click a suspicious/shortened link, and vague but official-sounding terminology (KYC).\n    *   Classification: PHISHING.\n\n4.  **Formulate the Reason:**\n    *   Needs to be one short reason.\n    *   Draft 1: It uses urgency and a suspicious shortened link to trick the user into clicking.\n    *   Draft 2: It creates false urgency and uses a suspicious shortened link to steal credentials.\n    *   Draft 3 (shorter): It uses urgency and a suspicious shortened link to trick the user.\n\n5.  **Format the Output:**\n    *   PHISHING: It uses urgency and a suspicious shortened link to trick the user.\n\n6.  **Review against constraints:**\n    *   Starts with PHISHING or SAFE? Yes (PHISHING).\n    *   Followed by a colon? Yes.\n    *   One short reason? Yes.\n    *   Word count of message was 15, links 1 (just confirming the input matches, not a constraint on my output).\n\n7.  **Final Polish:**\n    PHISHING: It uses urgency and a suspicious shortened link to trick the user into clicking."
                }
              ]
            },
            "response_metadata": {
              "model_name": "inclusionai/ling-3.0-flash",
              "id": "gen-1795776462-pU3s6h54PwtTuBnaHE7c",
              "created": 1795776462,
              "object": "chat.completion",
              "finish_reason": "stop",
              "logprobs": null,
              "model_provider": "openrouter",
              "cost": 3.64728e-05,
              "cost_details": {
                "upstream_inference_completions_cost": 3.5532e-05,
                "upstream_inference_prompt_cost": 9.408e-07,
                "upstream_inference_cost": 3.64728e-05
              }
            },
            "type": "ai",
            "id": "lc_run--01a21c9e-3b88-76d0-a8bb-b7d276c58191-0",
            "usage_metadata": {
              "input_tokens": 96,
              "output_tokens": 564,
              "total_tokens": 660,
              "input_token_details": {
                "cache_read": 64,
                "cache_creation": 0
              },
              "output_token_details": {
                "reasoning": 544
              }
            },
            "tool_calls": [],
            "invalid_tool_calls": []
          }
        }
      }
    ]
  ],
  "llm_output": {
    "model_name": "inclusionai/ling-3.0-flash",
    "id": "gen-1795776462-pU3s6h54PwtTuBnaHE7c",
    "created": 1795776462,
    "object": "chat.completion"
  },
  "run": null,
  "type": "LLMResult"
}
[chain/start] [chain:RunnableSequence > parser:StrOutputParser] Entering Parser run with input:
[inputs]
[chain/end] [chain:RunnableSequence > parser:StrOutputParser] [0ms] Exiting Parser run with output:
{
  "output": "PHISHING: It uses urgency and a suspicious shortened link to trick the user into clicking."
}
[chain/start] [chain:RunnableSequence > chain:format_output] Entering Chain run with input:
{
  "input": "PHISHING: It uses urgency and a suspicious shortened link to trick the user into clicking."
}
[chain/end] [chain:RunnableSequence > chain:format_output] [0ms] Exiting Chain run with output:
{
  "output": "Verdict: PHISHING\nReason: It uses urgency and a suspicious shortened link to trick the user into clicking.\nAdvice: Do not click any link. Delete the message and report it at cybercrime.gov.in or call 1930."
}
[chain/end] [chain:RunnableSequence] [3.57s] Exiting Chain run with output:
{
  "output": "Verdict: PHISHING\nReason: It uses urgency and a suspicious shortened link to trick the user into clicking.\nAdvice: Do not click any link. Delete the message and report it at cybercrime.gov.in or call 1930."
}
Verdict: PHISHING
Reason: It uses urgency and a suspicious shortened link to trick the user into clicking.
Advice: Do not click any link. Delete the message and report it at cybercrime.gov.in or call 1930.

"""