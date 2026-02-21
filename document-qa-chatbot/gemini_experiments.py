import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-2.5-flash')

print("="*60)
print("DAY 1 - GEMINI EXPERIMENTS")
print("="*60)
print("\nYou'll see 7 experiments showing how LLMs work!")
print("Press Enter after each experiment to continue...\n")
input("Press Enter to start...")

# Experiment 1: Simple Question
print("\n" + "="*60)
print("EXPERIMENT 1: Simple Question")
print("="*60)
print("\nPrompt: 'What is machine learning?'\n")
response = model.generate_content("What is machine learning?")
print(f"Response:\n{response.text}")
input("\n→ Press Enter for next experiment...")

# Experiment 2: Instruction Following
print("\n" + "="*60)
print("EXPERIMENT 2: Instruction Following")
print("="*60)
print("\nPrompt: 'Explain machine learning in exactly one sentence.'\n")
response = model.generate_content("Explain machine learning in exactly one sentence.")
print(f"Response:\n{response.text}")
print("\n💡 Notice: It followed the 'one sentence' instruction!")
input("\n→ Press Enter for next experiment...")

# Experiment 3: Audience Adaptation
print("\n" + "="*60)
print("EXPERIMENT 3: Audience Adaptation")
print("="*60)
print("\nPrompt: 'Explain machine learning to a 5-year-old child.'\n")
response = model.generate_content("Explain machine learning to a 5-year-old child.")
print(f"Response:\n{response.text}")
print("\n💡 Notice: Simpler language, uses analogies!")
input("\n→ Press Enter for next experiment...")

# Experiment 4: No Context (Zendria)
print("\n" + "="*60)
print("EXPERIMENT 4: No Context")
print("="*60)
print("\nPrompt: 'What is the capital of Zendria?'\n")
print("(Zendria is a made-up country)")
response = model.generate_content("What is the capital of Zendria?")
print(f"Response:\n{response.text}")
print("\n💡 Notice: LLM doesn't know (Zendria doesn't exist)")
input("\n→ Press Enter for next experiment...")

# Experiment 5: With Context (RAG Simulation!)
print("\n" + "="*60)
print("EXPERIMENT 5: With Context (RAG SIMULATION!)")
print("="*60)
print("\n🎯 THIS IS THE MOST IMPORTANT EXPERIMENT!")
print("\nNow we GIVE the LLM context about Zendria...\n")

prompt = """Based ONLY on the following context, answer the question.

Context: Zendria is a fictional country created for testing. Its capital city is Luminos, which was founded in 1823 and has a population of 5 million people. The city is known for its crystal towers.

Question: What is the capital of Zendria?

Answer:"""

response = model.generate_content(prompt)
print(f"Response:\n{response.text}")
print("\n💡 AMAZING! Now it knows because we provided context!")
print("💡 This is EXACTLY how RAG works!")
print("💡 We retrieve relevant info and give it to the LLM!")
input("\n→ Press Enter for next experiment...")

# Experiment 6: Temperature Effects
print("\n" + "="*60)
print("EXPERIMENT 6: Temperature Effects")
print("="*60)

print("\n🌡️ Temperature 0 (Deterministic - Same every time):\n")
config0 = genai.GenerationConfig(temperature=0)
model_temp0 = genai.GenerativeModel('gemini-2.5-flash', generation_config=config0)

prompt = "Write a creative one-sentence story about a robot."
for i in range(3):
    response = model_temp0.generate_content(prompt)
    print(f"Attempt {i+1}: {response.text}")

print("\n💡 Notice: All three are very similar!\n")
input("→ Press Enter to see Temperature 1...")

print("\n🌡️ Temperature 1 (Creative - Different every time):\n")
config1 = genai.GenerationConfig(temperature=1)
model_temp1 = genai.GenerativeModel('gemini-2.5-flash', generation_config=config1)

for i in range(3):
    response = model_temp1.generate_content(prompt)
    print(f"Attempt {i+1}: {response.text}")

print("\n💡 Notice: All three are VERY different!")
print("💡 For RAG chatbot: We'll use Temperature 0 (consistent answers)")
input("\n→ Press Enter for final experiment...")

# Experiment 7: Full RAG Simulation
print("\n" + "="*60)
print("EXPERIMENT 7: Full RAG Simulation")
print("="*60)
print("\n🎯 This simulates your COMPLETE chatbot!\n")

prompt = """Based on the following excerpts from our company handbook, answer the employee's question. Cite the section number in your answer.

Context:

Section 3.1: "All employees receive 15 days of paid vacation per year. Vacation days must be requested 2 weeks in advance through the HR portal."

Section 3.2: "Sick leave is separate from vacation time. Employees can take up to 10 days of sick leave per year without requiring a doctor's note."

Section 5.4: "Remote work is allowed up to 2 days per week with manager approval. Full-time remote work requires VP approval."

Question: How many vacation days do I get?

Answer:"""

response = model.generate_content(prompt)
print(f"Response:\n{response.text}")

print("\n" + "="*60)
print("🎉 ALL EXPERIMENTS COMPLETE!")
print("="*60)

print("\n📚 KEY LEARNINGS:")
print("\n1. ✅ LLMs follow instructions well")
print("2. ✅ They adapt tone/style based on audience")
print("3. ✅ WITHOUT context: They don't know your data")
print("4. ✅ WITH context: They answer accurately (RAG!)")
print("5. ✅ Temperature 0 = Consistent (for chatbots)")
print("6. ✅ Temperature 1 = Creative (for writing)")
print("7. ✅ Citations make answers trustworthy")

print("\n🚀 TOMORROW: You'll write CODE to do all this automatically!")
print("\n" + "="*60)
print("🎉 DAY 1 COMPLETE! CONGRATULATIONS!")
print("="*60)