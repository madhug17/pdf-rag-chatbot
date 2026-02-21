from langchain.text_splitter import RecursiveCharacterTextSplitter

document = """Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn from data, identify patterns, and make decisions without explicit programming. By utilizing statistical algorithms—such as linear regression or neural networks—ML improves performance as it consumes more data, driving applications like image recognition, recommendation engines, and predictive analytics. Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to learn from data, identify patterns, and make decisions without explicit programming. By utilizing statistical algorithms—such as linear regression or neural networks—ML improves performance as it consumes more data, driving applications like image recognition, recommendation engines, and predictive analytics."""

print("\nORIGINAL DOCUMENT")
print("-" * 60)
print(f"Total length: {len(document)} characters")
print(f"Estimated tokens: {len(document)//4}")
print("\nFirst 200 characters:")
print(document[:200])

splitter1 = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n", ".", " ", ""]
)

chunks = splitter1.split_text(document)

print(f"\nCreated {len(chunks)} chunks")
print("-" * 60)

for i, chunk in enumerate(chunks, 1):
    print(f"\nChunk {i} ({len(chunk)} chars)")
    print(chunk[:200] + "..." if len(chunk) > 150 else chunk)

if len(chunks) >= 2:
    print("\nLast 100 characters of Chunk 1:")
    print("..." + chunks[0][-100:])

    print("\nFirst 100 characters of Chunk 2:")
    print(chunks[1][:100] + "...")

print("\nEXPERIMENT 04")
print("-" * 60)

sizes = [300, 700, 1000]

for si in sizes:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=si,
        chunk_overlap=50
    )

    chunk_list = splitter.split_text(document)

    print(f"\nChunk size {si}:")
    print(f"Total chunks created: {len(chunk_list)}")
    print(f"Average chunk length: {sum(len(c) for c in chunk_list)//len(chunk_list)}")

print("\nEXPERIMENT 05")
print("-" * 60)
splitter_rag = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap=100
)
chunk_rag = splitter_rag.split_text(document)
user_question= "wht is supervised learning ?"
relevant_chunk=[]
for i, chunk in enumerate(chunk_rag,1):
    if "supervised learning" in chunk.lower():
        relevant_chunk.append((i,chunk))
for chunk_num, chunk in relevant_chunk:
    print(f"chunk {chunk_num}")
    print(chunk)

print("\n💡 In real RAG:")
print("   1. These chunks would be sent to LLM as context")
print("   2. LLM would use them to answer the question")
print("   3. LLM would cite which chunk the info came from")