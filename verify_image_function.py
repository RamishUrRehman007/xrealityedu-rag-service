import os
import sys
from dotenv import load_dotenv

# Ensure we can import from the directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieve_and_respond import answer_question, generate_cached_image

load_dotenv()

print("🧪 Testing Image Generation Logic...")

# Mock Data
student_name = "TestStudent"
grade = "Grade 10"
subject = "Biology"
topic = "Mitosis"

# 1. Direct Function Test
print(f"\n1. Direct Call to generate_cached_image('{topic}', '{grade}')...")
url = generate_cached_image(topic, grade)
if url:
    print(f"✅ Image Generated: {url}")
else:
    print("❌ Image Generation Failed")

# 2. Integration Test via answer_question
print(f"\n2. Testing answer_question with 'Show me an image of {topic}'...")
response = answer_question(
    question=f"Show me an image of {topic}",
    history="",
    subject=subject,
    student_name=student_name,
    grade_level=grade,
    mode="tutoring",
    current_topic=topic
)

print(f"\n🗣️ AI Response:\n{response}")

if "![" in response and url in response:
    print("\n✅ SUCCESS: Response contains the generated image!")
else:
    print("\n❌ FAILURE: Response missing image markdown.")
