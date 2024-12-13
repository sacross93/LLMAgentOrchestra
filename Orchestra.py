import os
from dotenv import load_dotenv
openaijykey = './.env'
load_dotenv(openaijykey)
gemini_api_key = os.getenv("GEMINI_API_KEY_JY")
os.environ['GOOGLE_API_KEY'] = gemini_api_key
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import typing_extensions as typing
import json
import textwrap
import pandas as pd

train_data = pd.read_csv('train.csv')

required_agents = f"""[배경]
자동차 산업의 빅데이터 활용을 활성화하고 신시장 창출 및 기업들의 문제 해결을 지원하기 위해 '2024 자동차 데이터 분석 경진대회'를 개최합니다.

이번 대회를 통해 구축된 자동차산업 오픈 생태계 플랫폼 데이터를 활용하여 자동차 산업의 다양한 현안을 해결하고, 혁신적인 신시장 창출의 가능성을 모색하며, 자동차 산업의 빅데이터 활용을 촉진하는 것을 목표로 하고 있습니다. 또한, 우수사례 발표를 통해 사업성과를 공유하고 확산함으로써 자동차 산업의 지속 가능한 성장에 기여하고자 합니다.

[주제]
프롬프트 엔지니어링을 통한 자동차 데이터 분류

[설명]
프롬프트 작성과 프롬프트 엔지니어링을 통해 자동차 관련 데이터를 분류해야합니다.

참가자는 system, user에 해당하는 프롬프트 제출 양식에 맞게 제출하면 평가 시스템에 연동된 GPT API를 통해 모델의 분류 성능과 프롬프트 토큰 길이를 바탕으로 정량적 평가 점수가 계산됩니다.

모델 : GPT3.5-turbo-0125
프롬프트 토큰 제한 : system + user prompt 기준, 16000 토큰까지
모델 출력 규칙 : 평가 데이터 40개 샘플들에 대해서 각 행 마다 예측 결과(0 또는 1)만을 출력
temperature : 0.4

[주의사항]
토큰을 적게 사용하고 평가 데이터 40개에 대해서 정확도가 높은 것을 목표로 합니다.

[출력 형식]
0,
1,
0,
1,
...

[Agent 구성]
1. 위의 작업을 잘 해야 합니다.
2. 프롬프트 엔지니어링을 통해 프롬프트를 작성해야 합니다.
3. 정확도가 90% 미만이거나 출력 형식이 올바르지 않다면 Agent를 통하여 프롬프트를 계속 개선해야 합니다.

[예시 데이터]
{train_data}
"""

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

# 프롬프트 작성 및 JSON 모드로 실행""
prompt = f"""You are the expert in determining how many LLM Agents are needed based on user queries and defining what each LLM Agent does.
There should be a reason why you need that many agents, and the name and role of each agent should be written down.
I would also like to see the end goal of the multi agents working together.

User Query: ```{required_agents}```
"""
result = model.generate_content(
    prompt,
    generation_config=genai.GenerationConfig(
        response_mime_type="application/json"
        # , response_schema=list[Recipe]
    ),
)

# Agent 정의 출력
# print(result.text)

result_json = json.loads(result.text)

print(result_json)


# LLMAgentOrchestra 클래스 정의
class LLMAgentOrchestra:
    def __init__(self, agent_template_manager):
        self.json_parser = JSONParser()
        self.agent_template_manager = agent_template_manager
        self.code_generator = LangGraphCodeGenerator(agent_template_manager)
        self.formatter = CodeFormatter()

    def create_multi_agent_system(self, agent_info_json, user_query):
        agent_info = self.json_parser.parse(agent_info_json)
        langgraph_code = self.code_generator.generate(agent_info, user_query)
        formatted_code = self.formatter.format(langgraph_code)
        return formatted_code

class JSONParser:
    def parse(self, agent_info_json):
        """
        JSON 문자열을 파싱하여 Python 객체로 변환합니다.

        Args:
            agent_info_json: JSON 형식의 Agent 정보

        Returns:
            파싱된 Agent 정보 (dict)
        """
        # JSON 문자열을 파싱하여 Python 객체로 변환
        # agent_info_str = agent_info_json.content # 이 코드는 주석처리
        try:
            agent_info = json.loads(agent_info_json)
            return agent_info
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

class AgentTemplateManager:
    def get_template(self, agent_type):
        """
        Agent 유형에 맞는 코드 템플릿을 반환합니다.

        Args:
            agent_type: Agent 유형 (str)

        Returns:
            Agent 코드 템플릿 (str)
        """
        # TODO: 다양한 Agent 유형에 대한 템플릿을 정의하고, agent_type에 따라 적절한 템플릿을 반환하도록 구현
        if agent_type == "Data Understanding Agent":
            return textwrap.dedent(
                """
                from langchain_core.messages import BaseMessage
                class DataUnderstandingAgent:
                    def __init__(self, description):
                        self.description = description

                    def run(self, messages: list[BaseMessage]) -> str:
                        # 사용자 질의를 분석하고 이해하는 로직을 구현합니다.
                        # 예: LLM을 사용하여 질의의 의도, 컨텍스트 등을 파악
                        user_query = messages[-1].content 
                        print(f"[DataUnderstandingAgent] Received query: {user_query}")

                        # 데이터 이해 결과 (예: 요약, 키워드 추출 등)
                        understanding = self.description

                        print(f"[DataUnderstandingAgent] Understanding: {understanding}")

                        return understanding
                """
            )
        elif agent_type == "Prompt Engineering Agent":
            return textwrap.dedent(
                """
                from langchain_core.messages import BaseMessage
                class PromptEngineeringAgent:
                    def __init__(self, description):
                        self.description = description

                    def run(self, messages: list[BaseMessage]) -> str:
                        # Data Understanding Agent의 결과를 바탕으로 프롬프트를 생성하는 로직을 구현합니다.
                        # 예: LLM을 사용하여 최적의 프롬프트 생성, 템플릿 기반 프롬프트 생성 등
                        understanding = messages[-1].content
                        print(f"[PromptEngineeringAgent] Received understanding: {understanding}")

                        # 생성된 프롬프트 (예: 사용자 질의 + 추가 지시 사항)
                        prompt = f"```{{understanding}}```\\n\\n주어진 데이터를 GPT3.5 모델을 활용해서 잘 분류해봐. system prompt와 user prompt의 토큰 수의 최대는 16000이하여야 해"

                        print(f"[PromptEngineeringAgent] Generated prompt: {prompt}")

                        return prompt
                """
            )
        elif agent_type == "Evaluation Agent":
            return textwrap.dedent(
                """
                from langchain_core.messages import BaseMessage
                class EvaluationAgent:
                    def __init__(self, description):
                        self.description = description

                    def run(self, messages: list[BaseMessage]) -> str:
                        # Prompt Engineering Agent의 프롬프트를 사용하여 모델을 실행하고 결과를 평가하는 로직을 구현합니다.
                        # 예: 모델 API 호출, 결과 파싱, 성능 지표 계산 등
                        prompt = messages[-1].content
                        print(f"[EvaluationAgent] Received prompt: {prompt}")

                        # 모델 실행 결과 (예: 분류 결과, 점수 등)
                        evaluation_result = "평가 결과: ..."

                        print(f"[EvaluationAgent] Evaluation result: {evaluation_result}")

                        return evaluation_result
                """
            )
        else:
            return textwrap.dedent(
                """
                class {agent_type}:
                    def __init__(self, description):
                        self.description = description
                        
                    def run(self, messages):
                        \"\"\"
                        Agent의 실행 로직을 구현합니다.
                        
                        Args:
                            messages: 이전 Agent로부터 전달받은 메시지
                            
                        Returns:
                            str: Agent의 실행 결과
                        \"\"\"
                        # TODO: Agent 역할에 맞는 실행 로직 구현
                        result = f"{agent_type} executed successfully." # 이 부분을 각 Agent에 맞게 수정
                        return result
                """
            ).format(agent_type=agent_type)

class LangGraphCodeGenerator:
    def __init__(self, agent_template_manager):
        self.agent_template_manager = agent_template_manager

    def generate(self, agent_info, user_query):
        """
        Agent 정보와 사용자 질의를 바탕으로 LangGraph 코드를 생성합니다.

        Args:
            agent_info: Agent 정보 (dict)
            user_query: 사용자 질의 (str)

        Returns:
            생성된 LangGraph 코드 (str)
        """
        agents_needed = agent_info.get("agents_needed", 0)
        agent_details = agent_info.get("agent_details", [])
        end_goal = agent_info.get("end_goal", "")

        agent_classes_code = ""
        agent_nodes_code = ""
        agent_edges_code = ""
        
        # 첫 번째 Agent를 entry point로 설정
        entry_point = agent_details[0]["agent_name"] if agent_details else None

        for i, agent_detail in enumerate(agent_details):
            agent_name = agent_detail["agent_name"]
            agent_role = agent_detail["agent_role"]

            # Agent 템플릿 가져오기
            agent_template = self.agent_template_manager.get_template(agent_name)

            # Agent 클래스 코드 생성
            agent_classes_code += textwrap.dedent(
                agent_template
            )
            agent_classes_code += "\n"

            # Agent 노드 추가 코드 생성
            agent_nodes_code += f"""
        {self.to_camel_case(agent_name)} = {agent_name}("{agent_role}")
        workflow.add_node("{agent_name}", {self.to_camel_case(agent_name)}.run)"""
            
            # Agent 간 엣지(의존성) 추가 코드 생성
            if i < len(agent_details) - 1:
                next_agent_name = agent_details[i + 1]["agent_name"]
                agent_edges_code += f"""
        workflow.add_edge("{agent_name}", "{next_agent_name}")"""

        # LangGraph 코드 템플릿
        langgraph_code_template = textwrap.dedent(
            """
        import asyncio
        from langchain_core.messages import BaseMessage
        from langchain_core.prompts import ChatPromptTemplate
        from langgraph.graph import StateGraph, END
        from typing import Dict, TypedDict, Annotated, Sequence
        import operator

        # Agent 클래스 정의
        {agent_classes_code}

        # State 정의
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]

        # Workflow 정의
        workflow = StateGraph(AgentState)
        
        # Agent 노드 추가
        {agent_nodes_code}

        # Agent 간 엣지 추가
        {agent_edges_code}

        # Entry point 설정
        workflow.set_entry_point("{entry_point}")

        # Workflow 컴파일
        app = workflow.compile()

        # 사용자 질의 메시지 생성
        user_query_message = [BaseMessage(content="{user_query}", type="user")]

        # Workflow 실행
        inputs = {{"messages": user_query_message}}
        result = asyncio.run(app.ainvoke(inputs))
        print(result)
        """
        )

        # LangGraph 코드 생성
        langgraph_code = langgraph_code_template.format(
            agent_classes_code=agent_classes_code,
            agent_nodes_code=agent_nodes_code,
            agent_edges_code=agent_edges_code,
            entry_point=entry_point,
            user_query=user_query,
        )

        return langgraph_code
    
    def to_camel_case(self, text):
        # 공백을 기준으로 문자열 분리
        words = text.split()
        # 첫 단어는 그대로 두고, 나머지 단어들의 첫 글자를 대문자로 변경
        camel_case_words = [words[0]] + [word.capitalize() for word in words[1:]]
        # 단어들을 다시 결합
        result = "".join(camel_case_words)
        return result

class CodeFormatter:
    def format(self, code):
        """
        코드를 포맷팅합니다. (여기서는 간단하게 들여쓰기만 조정)

        Args:
            code: 포맷팅할 코드 (str)

        Returns:
            포맷팅된 코드 (str)
        """
        formatted_code = textwrap.dedent(code)
        return formatted_code

# AgentTemplateManager 인스턴스 생성
agent_template_manager = AgentTemplateManager()

# LLMAgentOrchestra 인스턴스 생성
llm_agent_orchestra = LLMAgentOrchestra(agent_template_manager)

# LangGraph 코드 생성
generated_code = llm_agent_orchestra.create_multi_agent_system(result.text, required_agents) # result.text를 사용하도록 수정
print(generated_code)

