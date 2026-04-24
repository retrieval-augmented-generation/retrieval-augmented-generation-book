# We use gpt-3.5-turbo because it hallucinates more readily than newer models.
# The system prompt pushes it to answer instead of refusing, so we can see                                                                                                                               
# what a bare LLM produces when asked about a company it knows nothing about.
from dotenv import load_dotenv
load_dotenv()

import openai

client = openai.OpenAI()

system_prompt = """You are a knowledgeable enterprise software assistant.
Answer questions directly and specifically. Always provide concrete details
such as numbers, dates, and names when answering.
Keep every answer to one or two sentences."""

questions = [
    "How do I get my money back?",
    "What is the SLA uptime guarantee?",
    "How does the deployment pipeline work?",
    "How do I authenticate API requests?",
    "What does CVE-2024-1234 affect in Acme Corp's systems?",
]

for i, question in enumerate(questions, 1):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        max_tokens=80,
    )
    answer = response.choices[0].message.content
    print(f"Q{i}: {question}")
    print(f"A{i}: {answer}\n")
