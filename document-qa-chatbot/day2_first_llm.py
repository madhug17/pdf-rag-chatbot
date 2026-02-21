import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

print("=" * 60)
print("PROMPT TEMPLATES - EXAMPLE 1: Simple Template")
print("=" * 60)

template = """You are a helpful teacher.
Explain {topic} to a beginner in 2-3 sentences.
Use simple language and avoid jargon.
Explanation:"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=template
)

topics = ["RAG", "embeddings", "vector databases"]

for topic in topics:
    print(f"\nTopic: {topic}")
    print("-" * 60)

    formatted_prompt = prompt.format(topic=topic)
    response = llm.invoke(formatted_prompt)

    print(response.content)

print("\n" + "=" * 60)
print("EXAMPLE 2: Multiple Variables")
print("=" * 60)

template2 = """Explain {topic} as if talking to a {audience}.
Use appropriate language for that audience.
Explanation:"""

prompt2 = PromptTemplate(
    input_variables=["topic", "audience"],
    template=template2
)

combinations = [
    {"topic": "machine learning", "audience": "5-year-old child"},
    {"topic": "neural networks", "audience": "business executive"},
    {"topic": "transformers", "audience": "college student"}
]

for comb in combinations:
    print(f"\nTopic: {comb['topic']} | Audience: {comb['audience']}")
    print("-" * 60)

    formatted = prompt2.format(**comb)
    response = llm.invoke(formatted)

    print(response.content)

print("\n" + "=" * 60)
print("KEY LEARNING: Templates make prompts reusable!")
print("=" * 60)
