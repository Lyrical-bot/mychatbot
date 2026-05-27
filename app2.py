import streamlit as st
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

load_dotenv()

# Azure OpenAI 설정 로드
ENDPOINT = os.getenv("AZURE_OAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OAI_KEY")
DEPLOYMENT = os.getenv("AZURE_OAI_DEPLOYMENT")

async def call_openai_model(request_id, system_message, user_message, model, client):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    
    response = await client.chat.completions.create(
        model=model,
        temperature=0.7,
        max_tokens=800,
        messages=messages
    )
    return request_id, response.choices[0].message.content

async def run_multiple_requests(num_tasks, system_text, full_user_text, client):
    tasks = []
    for i in range(num_tasks):
        task_text = full_user_text + (f"\n(Request {i+1})" if num_tasks > 1 else "")
        tasks.append(call_openai_model(i + 1, system_text, task_text, DEPLOYMENT, client))
    
    return await asyncio.gather(*tasks)

st.set_page_config(page_title="Prompt Engineering Playground", layout="wide")
st.title("Prompt Engineering Service 🚀")

# 파일에서 기본값 읽기
def get_file_content(filepath, default_text):
    if os.path.exists(filepath):
        return open(filepath, encoding="utf8").read().strip()
    return default_text

with st.sidebar:
    st.header("Configuration")
    num_tasks = st.slider("Number of simultaneous tasks", 1, 5, 1)
    st.info(f"Deployment: {DEPLOYMENT}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Prompt Settings")
    system_input = st.text_area(
        "System Prompt (system.txt)", 
        value=get_file_content("system.txt", "You are a helpful assistant."),
        height=150
    )
    grounding_input = st.text_area(
        "Grounding Context (grounding.txt)", 
        value=get_file_content("grounding.txt", ""),
        height=250
    )

with col2:
    st.subheader("Chat Interface")
    user_message = st.text_input("Enter user message:", placeholder="질문을 입력하세요...")
    send_button = st.button("Send to Azure OpenAI", type="primary")

    if send_button:
        if not user_message:
            st.warning("메시지를 입력해주세요.")
        elif not all([ENDPOINT, API_KEY, DEPLOYMENT]):
            st.error("환경 변수(Endpoint, Key, Deployment)가 설정되지 않았습니다.")
        else:
            full_user_text = user_message + ("\n" + grounding_input if grounding_input else "")
            
            client = AsyncAzureOpenAI(
                azure_endpoint=ENDPOINT,
                api_key=API_KEY,
                api_version="2025-01-01-preview"
            )
            
            with st.spinner(f"{num_tasks}개의 응답을 생성 중..."):
                try:
                    results = asyncio.run(run_multiple_requests(num_tasks, system_input, full_user_text, client))
                    
                    for req_id, answer in results:
                        with st.chat_message("assistant"):
                            st.markdown(f"**Response {req_id}**")
                            st.write(answer)
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
