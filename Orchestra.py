import os
from dotenv import load_dotenv
openaijykey = './.env'
load_dotenv(openaijykey)
gemini_api_key = os.getenv("GEMINI_API_KEY_JY")
os.environ['GOOGLE_API_KEY'] = gemini_api_key
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import typing_extensions as typing

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

model = genai.GenerativeModel("gemini-2.0-flash-exp")

# JSON 스키마 정의
# class Recipe(typing.TypedDict):
#     recipe_name: str
#     ingredients: list[str]

# 프롬프트 작성 및 JSON 모드로 실행
prompt = "List a few popular cookie recipes."
result = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json"
        #, response_schema=list[Recipe]
    ),
)

print(result.text)