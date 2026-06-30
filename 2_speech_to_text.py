import os
import base64
import asyncio
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
import json

load_dotenv()

SOURCE_FOLDER = Path("./audios")
OUTPUT_FOLDER = Path("./transcribed_chunks")

MAX_CONCURRENT_TASKS = 1
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

class TranscriptSegment(BaseModel):
    tutorial_number: int = Field(
        description="The leading integer/number found at the very start of the video file name (e.g., 1 from '1_Installing VS Code...) "
    )
    video_title: str = Field(
        description="Original title of the tutorial video"
    )
    start: float = Field(
        description="Start timestamp in seconds where the spoken text begins"
    )

    end: float = Field(
        description="End timestamp in seconds where the spoken text ends"
    )

    text: str = Field(
        description="English transcription of the speech occurring between start and end timestamps"
    )

class Transcript(BaseModel):
    segments: List[TranscriptSegment] = Field(
        description="Ordered list of timestamped transcript segments"
    )


template = PromptTemplate(
    template='''
    You are an expert transcription assistant.
The user will provide the audio file name/title. Use this file metadata to populate the structural fields.

File name for this task: {{file_name}}

Tasks:
1. Transcribe the audio accurately.
2. If speech is Hindi, translate it into English.
3. Preserve meaning and context.
4. Return timestamped segments.
5. Extract the 'tutorial_number' strictly from the leading integer/number written at the very starting of the audio file name.
6. Extract the 'video_title' exactly as the remaining descriptive title of that video from the file name, ignoring the leading number and any file extensions. Every segment must carry these identical file metadata values

CRITICAL FORMATTING RULE: 
Return ONLY a raw, plain JSON object. Do NOT wrap your response in markdown code blocks (do not use ```json or ```). Start your response directly with the opening curly brace  and end with the closing curly brace .

Output schema:

{
  "segments":[
    {
      "start":0.0,
      "end":3.5,
      "text":"Today we begin the course."
    }
  ]
}

Rules:
- English only.
- No explanations.
- No markdown.
- Return valid JSON.
''',
input_variables=['file_name'],
template_format="jinja2"
)



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.0
)

transcription_model = model.with_structured_output(Transcript)


async def transcribe_file_async(mp3_path):
    async with semaphore:

        print(f"Processing: {mp3_path.name} (Slot acquired)")

        try:
            with open(mp3_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode("utf-8")

            formatted_prompt = template.invoke({'file_name': mp3_path.name}).to_string()

            message = HumanMessage(
            content=[
                {"type": "text", "text": formatted_prompt},
                {"type": "media", "mime_type": "audio/mp3", "data": audio_data}
                ]
            )

            max_retries = 4
            delay = 120

            for attempt in range(max_retries):
                try:
                        

                    full_transcript = await transcription_model.ainvoke([message])
                    
                    output_file_path = OUTPUT_FOLDER / f"{mp3_path.stem}.json"
                    with open(output_file_path, "w", encoding="utf-8") as out_file:
                        
                        json.dump(full_transcript.model_dump(), out_file, indent=2, ensure_ascii=False)
                    
                    print(f"Completed: {mp3_path.name}")
                    return
                
                except Exception as e:
                    if "503" in str(e) or "UNAVAILABLE" in str(e):
                        if attempt < max_retries - 1:
                            print(f"Server overloaded for {mp3_path.name}. Retrying in {delay}s... ")
                            await asyncio.sleep(delay)
                            #delay *= 2 
                        else:
                            raise e
                    else:
                        raise e 
            
        except Exception as e:
            print(f"Failed processing {mp3_path.name}: {str(e)}")

async def main():
    tasks = [transcribe_file_async(p) for p in SOURCE_FOLDER.glob("*.mp3")]
    print(f"Found {len(tasks)} files. Starting queue (Max {MAX_CONCURRENT_TASKS} at a time)...")
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
                