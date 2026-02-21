from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
c= canvas.Canvas("ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf",pagesize=letter)
c.setFont("Helvetica-Bold",16)
c.drawString(50,750,"machine learing guide")
c.setFont("Helvetica-Bold",12)
y = 700
texts=[
    "Machine learning is a method of data analysis that automates",
    "analytical model building. It uses algorithms that learn from",
    "data without being explicitly programmed.",
    "",
    "There are three main types:",
    "1. Supervised Learning - learns from labeled data",
    "2. Unsupervised Learning - finds patterns in unlabeled data",
    "3. Reinforcement Learning - learns through trial and error"
]
for text in texts:
    c.drawString(50,y,text)
    y-=20
c.showPage()
c.setFont("Helvetica",12)
y = 700
texts2 = [
    "Deep learning is a subset of machine learning that uses",
    "neural networks with multiple layers. These networks can",
    "learn complex patterns in large amounts of data.",
    "",
    "Applications include:",
    "- Image recognition",
    "- Natural language processing",
    "- Speech recognition",
    "- Autonomous vehicles"
]
for text in texts2:
    c.drawString(50,y,text)
    y-=20
c.save()
print(f"created: ML_CaseStudy_AIMLRhinos_EvenSem_2025-26.pdf")