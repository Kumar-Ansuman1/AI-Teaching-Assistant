import json
from pathlib import Path
from pydantic import BaseModel
from langchain_core.prompts import PromptTemplate
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

SOURCE_FOLDER = Path("./transcribed_chunks")
OUTPUT_FOLDER = Path("./semantic_chunks")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


class SemanticChunk(BaseModel):
    tutorial_number: int
    video_title: str
    topic: str
    start: float
    end: float
    text: str

class SemanticChunkOutput(BaseModel):
    chunks: List[SemanticChunk]


prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an expert at organizing programming tutorial transcripts.

You will receive a JSON transcript consisting of timestamped segments.

Your task is to merge consecutive segments into meaningful teaching chunks.

Rules:

1. Merge adjacent segments discussing the same programming concept.
2. Never split an explanation in the middle.
3. Start a new chunk only when the instructor switches to a new programming concept.
4. Preserve the transcript exactly.
5. Concatenate transcript text in chronological order.
6. Preserve the earliest start timestamp.
7. Preserve the latest end timestamp.
8. Generate a concise topic title.
9. Do NOT summarize or rewrite the transcript.
10. Return structured output only.

Output format:

{{
    "chunks":[
        {{
            "tutorial_number":1,
            "video_title":"...",
            "topic":"...",
            "start":0,
            "end":30,
            "text":"..."
        }}
    ]
}}

Transcript:

{transcript}
"""
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

structured_model = model.with_structured_output(SemanticChunkOutput)

chain = prompt | structured_model

files = sorted(SOURCE_FOLDER.glob("*.json"))

print(f"\nFound {len(files)} transcript files.\n")

success = 0
failed = 0

for file in files:
    try:
        print(f"Processing: {file.name}")

        with open(file, "r", encoding="utf-8") as f:
            transcript = json.load(f)
        
            result = chain.invoke(
            {
                "transcript": json.dumps(
                    transcript,
                    indent=2,
                    ensure_ascii=False
                )
            }
        )
        
        output_file = OUTPUT_FOLDER / file.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                result.model_dump(),
                f,
                indent=4,
                ensure_ascii=False
            )

        print(f"Saved: {output_file.name}")
        success += 1

    except Exception as e:
        print(f"Failed: {file.name}")
        print(f"Reason: {e}")
        failed += 1

print("\n" + "=" * 50)
print("Semantic Chunking Completed")
print("=" * 50)
print(f"Successful : {success}")
print(f"Failed     : {failed}")
print("=" * 50)