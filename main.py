import os
import yaml

from langchain_core.messages import HumanMessage, SystemMessage
import rag_system.vector_store
import rag_system.langchain_builder
import rag_system.database_handler

from munch import Munch

# test
def get_configs(path):
    assert os.path.exists(path), f"경로[{path}]에 해당 파일이 존재하지 않습니다. 프로그램을 종료합니다."
    with open(path) as f:
        config = yaml.safe_load(f)

    return config


def setting_rag_system(rag_cfg):
    vector_store = rag_system.vector_store.VectorStore(rag_cfg)
    vector_store.update()

    langchain_builder = rag_system.langchain_builder.LangChainBuilder(vector_store, rag_cfg)
    chain = langchain_builder.get_chain()

    return chain


def handling_database(rag_cfg):
    database_handler = rag_system.database_handler.DatabaseHandler(rag_cfg)
    update_datas = database_handler.check_new_data()
    if update_datas:
        database_handler.update_database(update_datas)
        print('-----Database updated-----')
    else:
        print('-----No new data-----')


def continual_chat(chain):
    print("AI와 대화를 시작하세요! 대화를 종료하려면 'exit'를 입력하세요.")
    chat_history = [] # 대화 기록을 수집 (메시지의 시퀀스)
    while True:
        query = input('User: ')

        if query == 'exit':
            break

        # 사용자의 질문을 검색 체인으로 처리
        result = chain.invoke({"input": query, "chat_history": chat_history})

        # AI의 답안을 출력
        print(f'AI: {result["answer"]}')

        # 대화 기록에 사용자의 질문과 AI의 답안을 추가
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result["answer"]))


if __name__ == '__main__':
    cfg = get_configs('config.yml')
    rag_cfg = Munch(cfg['rag'])

    handling_database(rag_cfg)
    chain = setting_rag_system(rag_cfg)
    continual_chat(chain)
    
