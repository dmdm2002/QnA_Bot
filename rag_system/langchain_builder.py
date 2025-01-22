from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class LangChainBuilder:
    def __init__(self, vector_store, rag_cfg):
        self.rag_cfg = rag_cfg
        self.llm = self.get_llm()
        self.vector_store = vector_store
        self.retriever = self.vector_store.query_vector_store()


    def get_llm(self):
        if self.rag_cfg.llm == 'openai':
            return ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=self.rag_cfg.api_key)
        elif self.rag_cfg.llm == 'ollama':
            return ChatOllama(model="llama3.1", temperature=0)
        else:
            raise ValueError(f"Unknown LLM: {self.rag_cfg.llm}")

    
    def set_history_aware_retriever(self):
        # 질문 재구성 프롬포트 템플릿 정의 및 생성과 히스토리 인식 검색기 생성
        contextualize_q_system_prompt = (
            "채팅 기록과 최신 사용자 질문이 주어졌을 때, "
            "해당 질문이 채팅 기록의 맥락을 참조할 수 있습니다. "
            "채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 재구성하세요. "
            "질문에 답변하지 말고, 필요한 경우에만 재구성하고 그렇지 않으면 그대로 반환하세요."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.retriever = create_history_aware_retriever(self.llm, self.retriever, contextualize_q_prompt)

        return self.retriever

    
    def self_question_answer_chain(self):
        qa_system_prompt = (
            "추가적인 정보가 필요하다면, 진단을 내리지말고 추가적인 질문을 하세요."
            "답을 모르면 모른다고만 말하세요."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        return question_answer_chain

    
    def get_chain(self):
        history_aware_retriever = self.set_history_aware_retriever()
        question_answer_chain = self.self_question_answer_chain()

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        return rag_chain
