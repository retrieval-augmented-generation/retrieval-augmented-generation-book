# bare_llm_test.py -- Ask domain-specific questions to a bare LLM (no retrieval).
from dotenv import load_dotenv
load_dotenv()
# Uses gpt-3.5-turbo which is more likely to hallucinate confidently.
# The system prompt encourages the model to answer rather than refuse.
import openai

client = openai.OpenAI()

system_prompt = """You are a knowledgeable enterprise software assistant.
Answer questions directly and specifically. Always provide concrete details
such as numbers, dates, and names when answering."""

questions = [
    "What is Acme Corp's refund policy for enterprise customers?",
    "When did Acme Corp change its pricing for the Pro tier?",
    "What are the SLA guarantees for Acme Corp's API uptime?",
    "Who approved the Q3 budget reallocation for the ML platform team?",
    "What security certifications does Acme Corp currently hold?",
]

for i, question in enumerate(questions, 1):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
    )
    answer = response.choices[0].message.content
    print(f"Q{i}: {question}")
    print(f"A{i}: {answer}\n")
