from dto import ChatbotRequest
import requests
import time
import logging

import os

from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain.utilities import GoogleSearchAPIWrapper

from langchain.memory import (
    ConversationBufferMemory,
    FileChatMessageHistory,
)


logger = logging.getLogger("Callback")

STEP10_1_PROMPT_TEMPLATE = "./prompts/10_project3/template_10_1_guess_intent.txt"
STEP10_2_PROMPT_TEMPLATE = "./prompts/10_project3/template_10_2_guess_chapter.txt"
STEP10_3_PROMPT_TEMPLATE = "./prompts/10_project3/template_10_3_explain_api.txt"
STEP10_4_PROMPT_TEMPLATE = "./prompts/10_project3/template_10_4_default.txt"
STEP10_5_PROMPT_TEMPLATE = "./prompts/10_project3/template_10_5_search_value_check.txt"
STEP10_6_PROMPT_TEMPLATE = "./prompts/10_project3/template_10_6_search_compress.txt"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 0

USE_RETRIEVER = True

intent_list = [
    "kakao_sync: Kakao sync API",
    "kakaotalk_channel: Kakaotalk channel API",
    "kakao_social: Kakao social API",
    "others: other questions"
]

DATA_PATH_DICT = {
    "kakao_sync": "../data/project_data_카카오싱크.txt",
    "kakaotalk_channel": "../data/project_data_카카오톡채널.txt",
    "kakao_social": "../data/project_data_카카오소셜.txt",
}

HISTORY_DIR = os.path.join(os.getcwd(), "chat_histories")

if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm: ChatOpenAI, template_path: str, output_key: str):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )


def get_chapter_list(intent: str):
    with open(DATA_PATH_DICT[intent], "r") as file:
        data = file.readlines()

    chapter_list = [text.replace("#", "").replace("\n", "").replace(" ", "") for text in data if text.startswith("#")]

    return chapter_list


def query_web_search(question: str, search_value_check_chain: ChatPromptTemplate, search_compression_chain: ChatPromptTemplate) -> str:
    context = {"question": question}
    context["related_web_search_results"] = search_tool.run(question)

    has_value = search_value_check_chain.run(context)

    if has_value == "Y":
        return search_compression_chain.run(context)
    else:
        return ""


def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    return FileChatMessageHistory(file_path)


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        chat_memory=history,
    )

    return memory.buffer


def search_db(intent: str, query: str, chunk_size: int=CHUNK_SIZE, chunk_overlap:int=CHUNK_OVERLAP, use_retriever: bool=True):
    def _prepare_data(intent: str, chunk_size: int, chunk_overlap: int):
        loader = TextLoader(DATA_PATH_DICT[intent])
        documents = loader.load()

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        return texts

    def _query_db(query: str, db, retriever, use_retriever: bool) -> list[str]:
        if use_retriever:
            docs = retriever.get_relevant_documents(query)
        else:
            docs = db.similarity_search(query)

        # 결과로 찾은 문서가 서로 포함관계인 경우, 길이가 더 긴 문서를 남긴다.
        stored_docs = [docs[0].page_content]
        for doc in docs[1:]:
            for stored_doc in stored_docs:
                if doc.page_content in stored_doc or \
                        stored_doc in doc.page_content:
                    store_doc = doc.page_content if len(doc.page_content) >= len(stored_doc) else stored_doc
                else:
                    stored_docs.append(doc.page_content)

        return "\n".join(stored_docs)

    texts = _prepare_data(
        intent=intent,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    db = Chroma.from_documents(
        texts,
        OpenAIEmbeddings(),
        collection_name="kakao_sync",
    )

    retriever = db.as_retriever()

    query_result = _query_db(
        query=query,
        db=db,
        retriever=retriever,
        use_retriever=USE_RETRIEVER,
    )

    return query_result


def generate_guide(question: str, llm: ChatOpenAI, conversation_id: str='fa1010') -> dict[str, str]:
    guess_intent_chain = create_chain(
        llm=llm,
        template_path=STEP10_1_PROMPT_TEMPLATE,
        output_key="intent",
    )

    guess_chapter_chain = create_chain(
        llm=llm,
        template_path=STEP10_2_PROMPT_TEMPLATE,
        output_key="chapter",
    )

    explain_api_chain = create_chain(
        llm=llm,
        template_path=STEP10_3_PROMPT_TEMPLATE,
        output_key="output",
    )

    default_chain = create_chain(
        llm=llm,
        template_path=STEP10_4_PROMPT_TEMPLATE,
        output_key="output"
    )

    search_value_check_chain = create_chain(
        llm=llm,
        template_path=STEP10_5_PROMPT_TEMPLATE,
        output_key="output",
    )

    search_compression_chain = create_chain(
        llm=llm,
        template_path=STEP10_6_PROMPT_TEMPLATE,
        output_key="output",
    )

    search = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY","AIzaSyDQG-gVG08O4rC2jgi2zoYHBt-SONJlESs"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID","02bd47ea92fc346c5")
    )

    search_tool = Tool(
        name="Google Search",
        description="Search Google for recent results.",
        func=search.run,
    )

    context = dict(
        question=question
    )
    context["input"] = context["question"]

    # intent 추출
    context["intent_list"] = "\n".join(intent_list)
    context["intent"] = guess_intent_chain.run(context).split(":")[0]

    # chapter 추출
    chapter_list = get_chapter_list(intent)
    context["chapter_list"] = "\n".join(chapter_list)
    context["chapter"] = guess_chapter_chain.run(context)

    # web search
    context["compressed_web_search_results"] = query_web_search(
        context["question"],
        search_value_check_chain=search_value_check_chain,
        search_compression_chain=search_compression_chain,
    )

    # chat memory
    history_file = load_conversation_history(conversation_id)
    context["chat_history"] = get_chat_history(conversation_id)

    if context["intent"] in ("kakao_sync", "kakaotalk_channel", "kakao_social"):
        context["related_documents"] = search_db(
            context["intent"],
            context["question"],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_retriever=USE_RETRIEVER,
        )
        answer = explain_api_chain.run(context)

    else:
        answer = default_chain.run(context)

    log_user_message(history_file, question)
    log_bot_message(history_file, answer)
    return answer


def prompt_with_langchain(query, model="gpt-3.5-turbo", temperature=0.0, max_tokens=512, use_functions=True):
    kakao_expert_llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    guide = generate_guide(
        question=query,
        llm=kakao_expert_llm,
        conversation_id='dsds'
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