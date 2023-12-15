from dto import ChatbotRequest
import requests
import time
import logging
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import SequentialChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

import requests

logger = logging.getLogger("Callback")

DATA_PATH = "../data/project_data_카카오싱크.txt"
STEP5_1_PROMPT_TEMPLATE = "./prompts/5_callback/template_5_1_knowledge_level.txt"
STEP5_2_PROMPT_TEMPLATE = "./prompts/5_callback/template_5_2_guide.txt"


def create_chain(llm, template_path, output_key):
    def _read_prompt_template(file_path: str) -> str:
        with open(file_path, "r") as f:
            prompt_template = f.read()

        return prompt_template

    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=_read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )


def generate_guide(question, llm) -> dict[str, str]:
    knowledge_level_chain = create_chain(
        llm,
        STEP5_1_PROMPT_TEMPLATE,
        "knowledge_level",
    )

    guide_chain = create_chain(
        llm,
        STEP5_2_PROMPT_TEMPLATE,
        "output",
    )

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

    # run
    context["question"] = question
    context = guide_chain(context)

    return context["output"]


def search_db(query: str):
    def _query_db(query: str, use_retriever: bool = False) -> list[str]:
        if use_retriever:
            docs = _retriever.get_relevant_documents(query)
        else:
            docs = _db.similarity_search(query)

        str_docs = [doc.page_content for doc in docs]
        return str_docs

    loader = TextLoader(DATA_PATH)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)

    _db = Chroma.from_documents(
        texts,
        OpenAIEmbeddings(),
        collection_name="kakao_sync",
    )

    _retriever = _db.as_retriever()

    query_result = _query_db(
        query=query,
        use_retriever=True,
    )

    search_results = []
    for document in query_result:
        search_results.append(
            {
                "content": document.split(':')[1]
            }
        )

    return search_results


def prompt_with_langchain(query, model="gpt-3.5-turbo", temperature=0.0, max_tokens=512, use_functions=True):
    kakao_expert_llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    guide = generate_guide(
        question=query,
        llm=kakao_expert_llm,
    )
    if use_functions:
        print("functions 사용 - agent를 생성해 문제에 응답합니다")

        functions = [
            {
                "name": "search_db",
                "func": lambda x: search_db(guide),
                "description": "Function to search extra information related to query from the database of Kakao sync"
            }
        ]

        tools = [
            Tool(
                **func
            ) for func in functions
        ]

        agent = initialize_agent(
            tools,
            kakao_expert_llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )

        result = agent.run(
            query
        )

    else:
        print("functions 미사용 - multi chain 으로 문제에 응답합니다.")

        result = guide

    return result


def callback_handler(request: ChatbotRequest) -> dict:
    # ===================== start =================================
    output_text = prompt_with_langchain(
        query=request.userRequest.utterance,
        use_functions=True,
        max_tokens=2048,
    )

    print(output_text)

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
        requests.post(url, json=payload)