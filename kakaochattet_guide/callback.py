from dto import ChatbotRequest
import aiohttp
import time
import logging
import openai

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import os

# audience_knowledge_level = "moderate"

# 환경 변수 처리 필요!
# openai.api_key = ''
# SYSTEM_MSG = f"""
# You are a provide of Kakao service.
#
# - Role: Explain about Kakao API
# - Audience's language: Korean
# - Audience's knowledge level: {audience_knowledge_level}
# """
logger = logging.getLogger("Callback")

def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )

def generate_guide(question) -> dict[str, str]:

    STEP4_1_PROMPT_TEMPLATE = "./prompts/4_langchain_chain/template_4_1_knowledge_level.txt"
    STEP4_2_PROMPT_TEMPLATE = "./prompts/4_langchain_chain/template_4_2_guide.txt"

    kakao_expert_llm = ChatOpenAI(temperature=0.1, max_tokens=500, model="gpt-3.5-turbo")

    knowledge_level_chain = create_chain(kakao_expert_llm, STEP4_1_PROMPT_TEMPLATE, "knowledge_level")

    guide_chain = create_chain(kakao_expert_llm, STEP4_2_PROMPT_TEMPLATE, "output")

    preprocess_chain = SequentialChain(
        chains=[
            knowledge_level_chain,
            guide_chain
        ],
        input_variables=["question"],
        output_variables=["knowledge_level"],
        verbose=True,
    )

    context = dict(
        question=question
    )
    context = preprocess_chain(context)

    context["question"] = question

    context = guide_chain(context)

    return context["output"]


async def callback_handler(request: ChatbotRequest) -> dict:

    # ===================== start =================================
    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_MSG},
    #         {"role": "user", "content": request.userRequest.utterance},
    #     ],
    #     temperature=0,
    # )
    # # focus
    # output_text = response.choices[0].message.content

    output_text = generate_guide(request.userRequest.utterance)

   # 참고링크 통해 payload 구조 확인 가능
    payload = {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": output_text
                    }
                }
            ]
        }
    }
    # ===================== end =================================
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/ai_chatbot_callback_guide
    # 참고링크1 : https://kakaobusiness.gitbook.io/main/tool/chatbot/skill_guide/answer_json_format

    time.sleep(1.0)

    url = request.userRequest.callbackUrl

    if url:
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url, json=payload, ssl=False) as resp:
                await resp.json()