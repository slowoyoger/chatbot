import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

from gtts import gTTS
import os

#파일 따로 따로 올리고 + 비디오 등록하고 검색가능한거
video_files = {}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_txt(txt_docs):
    text = ""
    for txt_file in txt_docs:
        content = txt_file.read()
        if isinstance(content, bytes):
            text += content.decode('utf-8')
        else:
            text += content
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    if vectorstore is not None:
        retriever = vectorstore.as_retriever()
    else:
        retriever = None
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    if retriever is not None:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory
        )
        return conversation_chain
    else:
        return None

# 채팅 메시지
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# 음성 메시지            
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

   # 'temp' 폴더 생성
    os.makedirs("temp", exist_ok=True)

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 != 0:
            chatbot_response = message.content  # 대화 히스토리의 메시지 내용을 chatbot_response에 저장
            user_message = user_template.replace("{{MSG}}", chatbot_response)
            st.write(user_message, unsafe_allow_html=True)

            # 음성으로 변환
            tts = gTTS(chatbot_response, lang='ja', slow=False)
            tts_file_path = "temp/tts_response.mp3"
            tts.save(tts_file_path)

            # 재생 버튼 추가
            st.audio(open(tts_file_path, 'rb').read(), format="audio/mp3", start_time=0)

            # 임시 파일 삭제
            os.remove(tts_file_path)
            
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    # load_dotenv()
    st.set_page_config(page_title="チャットボット - トヨ",
                       page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("トヨタのチャットボット「トヨ」です。 :robot_face:")
    user_question = st.chat_input("登録した書類について気になる点を質問してください。")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.title("書類アップロード")
        pdf_docs = st.file_uploader(
            "PDFファイルをアップロードしてください。", accept_multiple_files=True, type=["pdf"])

        txt_docs = st.file_uploader(
            "TXTファイルをアップロードしてください。", accept_multiple_files=True, type=["txt"])

        if st.button("登録"):
            with st.spinner("登録中"):
                pdf_raw_text = get_pdf_text(pdf_docs)
                txt_raw_text = get_text_from_txt(txt_docs)
                raw_text = pdf_raw_text + txt_raw_text
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    with st.sidebar:
        st.title("動画ガイドをアップロード・検索")
        

        
        uploaded_videos = st.file_uploader("動画をアップロードしてください。", accept_multiple_files=True, type=["mp4", "avi"])

        if uploaded_videos:
            st.subheader("アップロードされた動画ガイドリスト")
            for uploaded_video in uploaded_videos:
                video_name = uploaded_video.name
                video_files[video_name] = uploaded_video
                st.success(f"{video_name}がアップロードされました。")

            # 동영상 검색
            search_query = st.text_input("ガイドのタイトルを入力してください。")
            if search_query:
                results = [name for name in video_files.keys() if search_query.lower() in name.lower()]

                if results:
                    selected_video = st.selectbox("検索の結果", results)
                    st.success(f"検索された動画ガイド：{selected_video}")

                    # 동영상 재생
                    st.video(video_files[selected_video])

if __name__ == '__main__':
    main()
