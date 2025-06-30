__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import json
import re
import os
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import time
import torch
from datetime import datetime
import random
import numpy as np
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 페이지 설정 ---
st.set_page_config(page_title="임베딩 통합 벤치마크", layout="wide", page_icon="🚀")
st.title("🚀 임베딩 모델 통합 벤치마크 시스템")

# GPU 체크
device = "cuda" if torch.cuda.is_available() else "cpu"
st.info(f"사용 중인 디바이스: {device} 🎮" if device == "cuda" else f"사용 중인 디바이스: {device} 💻")

# --- 설정 ---
CHROMA_DB_PATH = r"C:\auto_excel\chromadb"

AVAILABLE_MODELS = {
    "jhgan/ko-sroberta-multitask": "naver_blogs_sroberta_768d",
    "BM-K/KoSimCSE-roberta": "naver_blogs_kosimcse_768d", 
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": "naver_blogs_minilm_384d",
    "sentence-transformers/distiluse-base-multilingual-cased-v1": "naver_blogs_distiluse_512d",
    "intfloat/multilingual-e5-base": "naver_blogs_e5base_768d",
    "BAAI/bge-large-en-v1.5": "naver_blogs_bge_large_en_1024d",
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": "naver_blogs_krsbert_768d",
    "intfloat/multilingual-e5-large": "naver_blogs_e5_large_1024d"
}

# --- 탭 생성 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📁 데이터 준비", "🤖 LLM 정확한 벤치마크", "🔧 컬렉션 구축", "📊 벤치마크 실행", "📈 성능 분석"])

# --- 헬퍼 함수들 ---
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

@st.cache_resource
def load_embedding_model(model_name):
    with st.spinner(f"모델 로딩 중: {model_name}..."):
        return SentenceTransformer(model_name, device=device)

@st.cache_resource  
def load_reranker_model(model_name="BAAI/bge-reranker-v2-m3"):
    with st.spinner(f"Re-ranker 모델 로딩 중: {model_name.split('/')[-1]}..."):
        return CrossEncoder(model_name, device=device)

def build_vector_database(json_data, model_name, collection_name, chunk_size=500, chunk_overlap=50):
    """JSON 데이터로부터 ChromaDB 컬렉션을 구축"""
    items = json_data.get('items', json_data)
    
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    chroma_client = get_chroma_client()
    
    try:
        chroma_client.delete_collection(name=collection_name)
        st.info(f"기존 '{collection_name}' 컬렉션을 삭제했습니다.")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func,
        metadata={"description": f"{model_name} 임베딩 사용 (청킹 적용)"}
    )
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    ids, documents, metadatas = [], [], []
    progress_bar = st.progress(0, text="청킹 진행 중...")
    
    for i, item in enumerate(items):
        original_id = str(item.get('id', f"item_{i}"))
        title = re.sub('<.*?>', '', str(item.get('title', '')))
        content = re.sub('<.*?>', '', str(item.get('description', '')))
        full_content = title + "\n\n" + content
        
        chunks = text_splitter.split_text(full_content)
        
        for j, chunk in enumerate(chunks):
            chunk_id = f"{original_id}_chunk_{j}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "original_id": original_id,
                "chunk_number": j,
                "title": title,
                "url": item.get('link', ''),
                "date": item.get('postdate', ''),
            })
        
        progress_bar.progress((i + 1) / len(items), text=f"청킹 진행 중... ({i+1}/{len(items)})")
    
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        st.success(f"✅ 총 {len(items)}개 문서에서 {len(ids)}개의 조각을 생성하여 '{collection_name}'에 저장완료!")
    
    return collection

def calculate_detailed_metrics(ranked_docs, relevant_docs, k=10):
    """상세 메트릭 계산 (MRR, NDCG 등)"""
    # MRR 계산
    mrr = 0.0
    for i, doc_id in enumerate(ranked_docs):
        if any(doc_id.startswith(rel + "_") for rel in relevant_docs):
            mrr = 1.0 / (i + 1)
            break
    
    # NDCG 계산
    dcg = 0.0
    for i, doc_id in enumerate(ranked_docs[:k]):
        if any(doc_id.startswith(rel + "_") for rel in relevant_docs):
            dcg += 1.0 / np.log2(i + 2)
    
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_docs), k)))
    ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    # Precision@K
    precision_at_k = sum(
        1 for doc_id in ranked_docs[:k] 
        if any(doc_id.startswith(rel + "_") for rel in relevant_docs)
    ) / k
    
    return {"mrr": mrr, "ndcg": ndcg, "precision_at_k": precision_at_k}

# --- 그래프 생성 함수들 ---
def create_model_comparison_chart(benchmark_results):
    """모델별 성능 비교 차트 - 오류 처리 개선"""
    df = pd.DataFrame(benchmark_results)
    
    if df.empty:
        st.error("❌ 벤치마크 결과가 없습니다.")
        return None
    
    # Hit Rate 컬럼 찾기
    hit_rate_cols = [col for col in df.columns if "Hit Rate" in col]
    if not hit_rate_cols:
        st.error("❌ Hit Rate 컬럼을 찾을 수 없습니다.")
        return None
    
    hit_rate_col = hit_rate_cols[0]
    
    # 오류 문자열 안전하게 처리
    def safe_convert_to_float(value):
        try:
            if isinstance(value, str):
                if '오류' in value or 'Error' in value:
                    return 0.0
                clean_value = value.replace('%', '').replace('+', '').strip()
                return float(clean_value)
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    df['hit_rate_numeric'] = df[hit_rate_col].apply(safe_convert_to_float)
    df['model_short'] = df['Model'].apply(lambda x: x.split('/')[-1] if isinstance(x, str) else str(x))
    
    # 유효한 결과만 필터링
    valid_df = df[df['hit_rate_numeric'] > 0]
    
    if len(valid_df) == 0:
        st.error("⚠️ 유효한 벤치마크 결과가 없습니다. 모든 모델에서 오류 발생")
        return None
    
    fig = px.bar(
        valid_df, 
        x='model_short', 
        y='hit_rate_numeric',
        color='hit_rate_numeric',
        color_continuous_scale='Viridis',
        title='📊 모델별 Hit Rate 성능 비교',
        labels={'hit_rate_numeric': 'Hit Rate (%)', 'model_short': '모델명'}
    )
    
    # 올바른 Plotly 문법
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        showlegend=False
    )
    
    return fig

def create_reranker_comparison_chart(results_with_without_reranker):
    """Re-ranker 사용 전후 비교 차트"""
    if not results_with_without_reranker:
        return None
        
    df = pd.DataFrame(results_with_without_reranker)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Re-ranker 미사용',
        x=df['Model'],
        y=df['Without_Reranker'],
        marker_color='lightcoral'
    ))
    
    fig.add_trace(go.Bar(
        name='Re-ranker 사용',
        x=df['Model'],
        y=df['With_Reranker'],
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title='🔄 Re-ranker 사용 전후 성능 비교',
        xaxis_title='모델명',
        yaxis_title='Hit Rate (%)',
        barmode='group',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def create_k_value_analysis_chart(k_analysis_results):
    """K값에 따른 성능 변화 분석"""
    fig = go.Figure()
    
    for model, results in k_analysis_results.items():
        fig.add_trace(go.Scatter(
            x=results['k_values'],
            y=results['hit_rates'],
            mode='lines+markers',
            name=model.split('/')[-1],
            line=dict(width=3),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title='📈 K값에 따른 Hit Rate 변화',
        xaxis_title='K 값',
        yaxis_title='Hit Rate (%)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_question_type_analysis_chart(question_type_results):
    """질문 유형별 성능 분석"""
    fig = px.box(
        question_type_results,
        x='question_type',
        y='hit_rate',
        color='question_type',
        title='🎯 질문 유형별 성능 분포',
        labels={'question_type': '질문 유형', 'hit_rate': 'Hit Rate (%)'}
    )
    
    fig.update_layout(height=500)
    return fig

def create_detailed_metrics_radar_chart(detailed_metrics):
    """상세 메트릭 레이더 차트"""
    if not detailed_metrics:
        return None
        
    fig = go.Figure()
    
    metrics = ['Hit Rate', 'MRR', 'NDCG', 'Precision@K']
    
    for model, values in detailed_metrics.items():
        model_short = model.split('/')[-1]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=model_short
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title='🕸️ 모델별 종합 성능 레이더 차트',
        height=500
    )
    
    return fig

# 디버깅을 위한 상세 벤치마크 함수
def debug_benchmark_performance(model_name, collection_name, benchmark_data, sample_size=5):
    """벤치마크 성능 디버깅"""
    try:
        retriever_model = load_embedding_model(model_name)
        chroma_client = get_chroma_client()
        collection = chroma_client.get_collection(name=collection_name)
        
        st.write(f"🔍 **{model_name.split('/')[-1]} 디버깅**")
        st.write(f"- 컬렉션: {collection_name}")
        st.write(f"- 총 문서 수: {collection.count()}")
        
        # 샘플 질문들로 테스트
        sample_items = benchmark_data[:sample_size]
        
        for i, item in enumerate(sample_items):
            st.write(f"\n**질문 {i+1}**: {item['query']}")
            st.write(f"**정답 문서**: {item['relevant_docs']}")
            
            # 검색 실행
            query_embedding = retriever_model.encode(item['query']).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=10,
                include=["metadatas"]
            )
            
            retrieved_ids = results['ids'][0]
            st.write(f"**검색된 상위 10개**: {retrieved_ids}")
            
            # 청킹된 ID 매칭 확인
            relevant_docs = set(item['relevant_docs'])
            matches = []
            
            for r_id in retrieved_ids:
                for ans_id in relevant_docs:
                    if r_id.startswith(ans_id + "_"):
                        matches.append(r_id)
                        break
            
            if matches:
                st.success(f"✅ **매칭 성공**: {matches}")
            else:
                st.error(f"❌ **매칭 실패**: 정답과 일치하는 문서를 찾지 못함")
                
                # 실제 컬렉션에 있는 ID들 샘플 확인
                all_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=50
                )
                all_ids = all_results['ids'][0]
                
                # 정답 ID로 시작하는 청크들이 있는지 확인
                answer_chunks = []
                for ans_id in relevant_docs:
                    matching_chunks = [chunk_id for chunk_id in all_ids if chunk_id.startswith(ans_id + "_")]
                    answer_chunks.extend(matching_chunks)
                
                if answer_chunks:
                    st.warning(f"⚠️ **정답 청크들이 존재하지만 상위 10개에 없음**: {answer_chunks[:5]}")
                else:
                    st.error(f"💥 **정답 청크들이 컬렉션에 아예 없음**. 벤치마크 데이터와 컬렉션 불일치!")
            
            st.write("---")
            
    except Exception as e:
        st.error(f"디버깅 중 오류 발생: {e}")

def check_collection_health():
    """컬렉션 상태 및 데이터 일관성 확인"""
    try:
        chroma_client = get_chroma_client()
        collections = chroma_client.list_collections()
        
        st.subheader("🏥 컬렉션 건강 상태 체크")
        
        health_data = []
        for collection_info in collections:
            collection = chroma_client.get_collection(collection_info.name)
            count = collection.count()
            
            # 샘플 데이터 확인
            sample = collection.get(limit=5, include=['ids', 'metadatas'])
            sample_ids = sample.get('ids', [])
            
            # ID 패턴 확인 (청킹되었는지)
            chunked_ids = [id for id in sample_ids if '_chunk_' in id]
            chunked_ratio = len(chunked_ids) / len(sample_ids) * 100 if sample_ids else 0
            
            health_data.append({
                'Collection': collection_info.name,
                'Document Count': count,
                'Sample IDs': ', '.join(sample_ids[:3]),
                'Chunked Ratio': f"{chunked_ratio:.1f}%",
                'Status': '✅ 정상' if count > 0 and chunked_ratio > 0 else '⚠️ 문제'
            })
        
        health_df = pd.DataFrame(health_data)
        st.dataframe(health_df, use_container_width=True)
        
        return health_df
        
    except Exception as e:
        st.error(f"컬렉션 상태 체크 실패: {e}")
        return None

# --- TAB 1: 데이터 준비 ---
with tab1:
    st.header("📁 데이터 소스 선택")
    
    data_source = st.radio("데이터 소스를 선택하세요:", 
                          ["로컬 파일 경로", "파일 업로드"])
    
    json_data = None
    
    if data_source == "로컬 파일 경로":
        file_path = st.text_input("JSON 파일 경로를 입력하세요:", 
                                 value=r"C:\auto_excel\naver_blog_모기용품_20250617_131814.json")
        
        if st.button("파일 로드") and file_path:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    st.session_state['json_data'] = json_data
                    items = json_data.get('items', json_data)
                    st.success(f"✅ 파일 로드 성공! 총 {len(items)}개 문서")
                    
                    if items:
                        st.write("**데이터 미리보기:**")
                        preview_item = items[0]
                        st.json({k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                                for k, v in preview_item.items()})
                        
                except Exception as e:
                    st.error(f"파일 로드 실패: {e}")
            else:
                st.error("파일을 찾을 수 없습니다.")
    
    else:
        uploaded_file = st.file_uploader("JSON 파일을 업로드하세요:", type=['json'])
        
        if uploaded_file:
            try:
                json_data = json.load(uploaded_file)
                st.session_state['json_data'] = json_data
                items = json_data.get('items', json_data)
                st.success(f"✅ 파일 업로드 성공! 총 {len(items)}개 문서")
                
                if items:
                    st.write("**데이터 미리보기:**")
                    preview_item = items[0]
                    st.json({k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                            for k, v in preview_item.items()})
                    
            except Exception as e:
                st.error(f"파일 파싱 실패: {e}")

# --- TAB 2: LLM 기반 정확한 벤치마크 생성 ---
with tab2:
    st.header("🤖 LLM 기반 정확한 벤치마크 생성")
    st.success("🎯 **추천 방식**: 실제 문서 내용을 읽고 정확한 질문-정답 쌍을 생성합니다")
    
    # OpenAI API 키 입력
    col1, col2 = st.columns([2, 1])
    
    with col1:
        api_key = st.text_input(
            "OpenAI API 키", 
            type="password",
            help="gpt-4o-mini 모델 사용 (비용: ~$0.01/100질문)"
        )
    
    with col2:
        if st.button("🔍 API 키 테스트"):
            if api_key:
                try:
                    from openai import OpenAI
                    test_client = OpenAI(api_key=api_key)
                    test_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": "테스트"}],
                        max_tokens=5
                    )
                    st.success("✅ API 키 유효")
                except Exception as e:
                    st.error(f"❌ API 키 오류: {str(e)[:50]}")
            else:
                st.warning("API 키를 입력하세요")
    
    if not api_key or "sk-" not in api_key:
        st.warning("⬆️ 먼저 OpenAI API 키를 입력해주세요")
        
        with st.expander("💡 OpenAI API 키 발급 방법"):
            st.markdown("""
            1. **https://platform.openai.com** 접속
            2. **API keys** 메뉴 클릭
            3. **Create new secret key** 클릭
            4. 생성된 키를 복사해서 위에 입력
            
            💰 **비용**: gpt-4o-mini 모델 사용 시 100개 질문당 약 $0.01
            """)
    
    else:
        # LLM 벤치마크 생성 인터페이스
        st.subheader("⚙️ 생성 설정")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 사용 가능한 컬렉션 확인
            try:
                chroma_client = get_chroma_client()
                collections = chroma_client.list_collections()
                if collections:
                    collection_options = []
                    for col in collections:
                        collection_obj = chroma_client.get_collection(col.name)
                        count = collection_obj.count()
                        collection_options.append((col.name, count))
                    
                    selected_collection = st.selectbox(
                        "소스 컬렉션 선택",
                        options=[name for name, count in collection_options],
                        format_func=lambda x: f"{x} ({dict(collection_options)[x]}개 문서)"
                    )
                else:
                    st.error("❌ 컬렉션이 없습니다. 먼저 '컬렉션 구축'을 해주세요.")
                    selected_collection = None
                    
            except Exception as e:
                st.error(f"컬렉션 확인 실패: {e}")
                selected_collection = None
        
        with col2:
            num_docs = st.number_input(
                "처리할 문서 수", 
                min_value=5, max_value=100, value=20, step=5,
                help="너무 많으면 API 비용이 증가합니다"
            )
            
            questions_per_doc = st.number_input(
                "문서당 질문 수", 
                min_value=1, max_value=3, value=2, step=1
            )
        
        with col3:
            question_types = st.multiselect(
                "질문 유형",
                ["사실 확인", "방법 설명", "이유 설명", "비교", "추천"],
                default=["사실 확인", "방법 설명", "이유 설명"],
                help="다양한 유형의 질문을 생성합니다"
            )
            
            output_filename = st.text_input(
                "출력 파일명",
                value=f"llm_benchmark_{datetime.now().strftime('%m%d_%H%M')}.json"
            )
        
        # 예상 정보 표시
        total_questions = num_docs * questions_per_doc
        estimated_cost = total_questions * 0.0001
        
        st.info(f"💰 **예상 비용**: ${estimated_cost:.4f} | **예상 질문 수**: {total_questions}개")
        
    
        
        # LLM 벤치마크 생성 실행
        if st.button("🚀 정확한 벤치마크 생성!", use_container_width=True, type="primary"):
            if not selected_collection:
                st.error("❌ 컬렉션을 선택해주세요")
            elif not question_types:
                st.error("❌ 질문 유형을 선택해주세요")
            else:
                # 실제 LLM 기반 생성 로직
                try:
                    from openai import OpenAI
                    openai_client = OpenAI(api_key=api_key)
                    
                    collection = chroma_client.get_collection(selected_collection)
                    
                    # 진행 상황 표시
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.info("문서 샘플링 중...")
                    
                    # 무작위로 문서 샘플링
                    all_data = collection.get(include=["documents", "metadatas"])
                    total_available = len(all_data['ids'])
                    
                    if total_available < num_docs:
                        st.warning(f"⚠️ 요청한 {num_docs}개보다 적은 {total_available}개 문서만 있습니다.")
                        num_docs = total_available
                    
                    # 랜덤 샘플링
                    sample_indices = random.sample(range(total_available), num_docs)
                    
                    generated_questions = []
                    
                    for i, idx in enumerate(sample_indices):
                        progress = (i / len(sample_indices)) * 0.9
                        progress_bar.progress(progress)
                        status_text.info(f"질문 생성 중... ({i+1}/{len(sample_indices)})")
                        
                        doc_id = all_data['ids'][idx]
                        content = all_data['documents'][idx]
                        metadata = all_data['metadatas'][idx]
                        
                        # 너무 짧은 문서는 건너뛰기
                        if len(content.strip()) < 50:
                            continue
                        
                        # 원본 문서 ID 추출 (청킹된 ID에서)
                        original_id = metadata.get('original_id', '_'.join(doc_id.split('_')[:2]))
                        title = metadata.get('title', '제목 없음')[:100]
                        
                        # LLM으로 질문 생성
                        type_prompts = {
                            "사실 확인": "이 문서에서 확인할 수 있는 구체적인 사실에 대한 질문",
                            "방법 설명": "이 문서에서 설명하는 방법이나 과정에 대한 질문",
                            "이유 설명": "이 문서에서 설명하는 이유나 원인에 대한 질문",
                            "비교": "이 문서에서 언급되는 비교나 차이점에 대한 질문",
                            "추천": "이 문서에서 추천하거나 제안하는 것에 대한 질문"
                        }
                        
                        selected_types = random.sample(question_types, min(len(question_types), questions_per_doc))
                        
                        prompt = f"""다음 블로그 글의 내용을 바탕으로 정확히 답변할 수 있는 자연스러운 질문을 만들어주세요.

[제목]: {title}
[내용]: {content[:800]}

**요구사항:**
1. 이 문서의 내용만으로 답변할 수 있는 질문만 만들기
2. 다음 유형의 질문 각 1개씩 만들기:
{chr(10).join(f"   - {type_prompts[t]}" for t in selected_types)}

**출력 형식:**
질문만 한 줄씩 작성해주세요. 번호는 붙이지 마세요.
총 {questions_per_doc}개의 질문을 만들어주세요."""

                        try:
                            response = openai_client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3,
                                max_tokens=300
                            )
                            
                            generated_text = response.choices[0].message.content.strip()
                            
                            for line in generated_text.split('\n'):
                                line = line.strip()
                                if line and len(line) > 10:
                                    # 번호 제거
                                    line = re.sub(r'^\d+[\.\)]\s*', '', line)
                                    line = re.sub(r'^[-\*]\s*', '', line)
                                    
                                    if not line.endswith('?'):
                                        line += '?'
                                    
                                    generated_questions.append({
                                        "query": line,
                                        "relevant_docs": [original_id],
                                        "question_type": "llm_generated",
                                        "source_title": title,
                                        "generation_method": "document_based"
                                    })
                            
                            time.sleep(0.5)  # API 제한 고려
                            
                        except Exception as e:
                            st.warning(f"문서 {i+1} 처리 실패: {str(e)[:50]}")
                            continue
                    
                    # 결과 저장
                    progress_bar.progress(0.95)
                    status_text.info("결과 저장 중...")
                    
                    if generated_questions:
                        # 중복 제거 및 품질 필터링
                        unique_questions = []
                        seen_queries = set()
                        
                        for q in generated_questions:
                            query_clean = q['query'].lower().strip()
                            if query_clean not in seen_queries and len(q['query']) > 15:
                                seen_queries.add(query_clean)
                                unique_questions.append(q)
                        
                        # 최종 저장
                        with open(output_filename, 'w', encoding='utf-8') as f:
                            json.dump(unique_questions, f, ensure_ascii=False, indent=2)
                        
                        progress_bar.progress(1.0)
                        status_text.success(f"✅ 완료! {len(unique_questions)}개 질문 생성됨")
                        
                        # 결과 요약
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("생성된 질문", len(unique_questions))
                        with col2:
                            st.metric("처리된 문서", len(sample_indices))
                        with col3:
                            actual_cost = len(unique_questions) * 0.0001
                            st.metric("실제 비용", f"${actual_cost:.4f}")
                        with col4:
                            avg_per_doc = len(unique_questions) / len(sample_indices)
                            st.metric("문서당 질문", f"{avg_per_doc:.1f}개")
                        
                        # 샘플 질문 표시
                        st.subheader("📝 생성된 질문 샘플")
                        
                        sample_qs = random.sample(unique_questions, min(3, len(unique_questions)))
                        for i, q in enumerate(sample_qs, 1):
                            with st.expander(f"샘플 질문 {i}: {q['query'][:40]}..."):
                                st.write(f"**질문**: {q['query']}")
                                st.write(f"**정답 문서**: {q['relevant_docs'][0]}")
                                st.write(f"**출처**: {q.get('source_title', 'N/A')}")
                        
                        # 다운로드 버튼
                        with open(output_filename, 'r', encoding='utf-8') as f:
                            file_content = f.read()
                        
                        st.download_button(
                            label="📥 LLM 벤치마크 다운로드",
                            data=file_content,
                            file_name=output_filename,
                            mime="application/json"
                        )
                        
                        st.success("""
                        🎯 **이제 '벤치마크 실행' 탭에서 이 파일을 사용하세요!**
                        
                        정확한 질문-정답 쌍으로 인해 훨씬 높고 신뢰할 수 있는 성능을 확인할 수 있습니다.
                        """)
                        
                    else:
                        st.error("❌ 질문 생성에 실패했습니다. API 키와 설정을 확인해주세요.")
                        
                except Exception as e:
                    st.error(f"❌ 생성 중 오류: {e}")

# --- TAB 3: 컬렉션 구축 ---
with tab3:
    st.header("🔧 ChromaDB 컬렉션 구축")
    
    if 'json_data' not in st.session_state:
        st.warning("먼저 '데이터 준비' 탭에서 JSON 데이터를 로드해주세요.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("청킹 설정")
            chunk_size = st.number_input("청크 크기", min_value=100, max_value=2000, value=500, step=100)
            chunk_overlap = st.number_input("청크 겹침", min_value=0, max_value=200, value=50, step=10)
        
        with col2:
            st.subheader("모델 선택")
            selected_models = st.multiselect(
                "구축할 모델들을 선택하세요:",
                options=list(AVAILABLE_MODELS.keys()),
                default=["jhgan/ko-sroberta-multitask", "intfloat/multilingual-e5-base"],
                help="여러 모델을 선택하면 각각 별도 컬렉션이 생성됩니다."
            )
        
        if st.button("🔧 컬렉션 구축 시작!", use_container_width=True, type="primary"):
            if selected_models:
                total_models = len(selected_models)
                main_progress = st.progress(0, text="컬렉션 구축 시작...")
                
                for i, model_name in enumerate(selected_models):
                    collection_name = AVAILABLE_MODELS[model_name]
                    st.subheader(f"[{i+1}/{total_models}] {model_name}")
                    
                    try:
                        build_vector_database(
                            st.session_state['json_data'], 
                            model_name, 
                            collection_name,
                            chunk_size,
                            chunk_overlap
                        )
                    except Exception as e:
                        st.error(f"❌ {model_name} 구축 실패: {e}")
                    
                    main_progress.progress((i + 1) / total_models, 
                                         text=f"진행 상황: {i+1}/{total_models} 완료")
                
                st.success("🎉 모든 컬렉션 구축 완료!")
            else:
                st.warning("최소 하나의 모델을 선택해주세요.")

# --- TAB 4: 벤치마크 실행 ---
with tab4:
    st.header("📊 벤치마크 실행")
    
    try:
        chroma_client = get_chroma_client()
        existing_collections = [col.name for col in chroma_client.list_collections()]
        available_models = {k: v for k, v in AVAILABLE_MODELS.items() if v in existing_collections}
        
        if not available_models:
            st.warning("사용 가능한 컬렉션이 없습니다. '컬렉션 구축' 탭에서 먼저 구축해주세요.")
        else:
            st.info(f"사용 가능한 컬렉션: {len(available_models)}개")
            st.table(pd.DataFrame(list(available_models.items()), columns=['Model', 'Collection']))
            
    except Exception as e:
        st.error(f"ChromaDB 연결 실패: {e}")
        available_models = {}
    
    if available_models:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("벤치마크 설정")
            k_value = st.slider("최종 상위 K개 결과", 1, 20, 5)
            use_reranker = st.checkbox("Re-ranking 사용", value=True)
            initial_k = st.number_input("1차 검색 개수", min_value=k_value, max_value=50, value=25, disabled=not use_reranker)
            use_batch = st.checkbox("배치 처리 사용", value=True, help="GPU 가속")
            calculate_detailed = st.checkbox("상세 메트릭 계산", value=True, help="MRR, NDCG 등")
        
        with col2:
            st.subheader("벤치마크 데이터")
            
            # LLM 생성 파일 자동 감지
            try:
                llm_files = [f for f in os.listdir('.') if f.startswith('llm_benchmark_') and f.endswith('.json')]
            except:
                llm_files = []
            
            if llm_files:
                st.success(f"🤖 LLM 생성 파일 발견: {len(llm_files)}개")
                selected_llm_file = st.selectbox("LLM 벤치마크 파일 선택:", llm_files)
                benchmark_file_path = selected_llm_file
            else:
                st.warning("💡 LLM 생성 파일이 없습니다. 'LLM 정확한 벤치마크' 탭에서 먼저 생성하세요.")
                benchmark_file_path = st.text_input("벤치마크 파일 경로 (직접 입력)", value="")
            
            if benchmark_file_path and os.path.exists(benchmark_file_path):
                with open(benchmark_file_path, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)
                
                # 파일 유형 확인
                is_llm_generated = any(item.get('generation_method') == 'document_based' for item in benchmark_data)
                
                if is_llm_generated:
                    st.success(f"✅ LLM 정확한 벤치마크 로드됨 ({len(benchmark_data)}개 질문)")
                    st.info("🎯 **높은 정확도의 벤치마크 결과를 기대할 수 있습니다!**")
                else:
                    st.warning(f"⚠️ 기존 방식 벤치마크 로드됨 ({len(benchmark_data)}개 질문)")
                    st.info("💡 더 정확한 결과를 위해 LLM 벤치마크를 사용해보세요.")
                
                if benchmark_data:
                    type_counts = {}
                    difficulty_counts = {}
                    for item in benchmark_data:
                        q_type = item.get('question_type', 'unknown')
                        difficulty = item.get('difficulty', 'basic')
                        type_counts[q_type] = type_counts.get(q_type, 0) + 1
                        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                    
                    st.write("**질문 유형 분포:**")
                    for q_type, count in type_counts.items():
                        emoji = "🤖" if q_type == "llm_generated" else "🔥" if q_type in ['comparison', 'reasoning', 'complex'] else "📝"
                        st.write(f"- {emoji} {q_type}: {count}개")
                    
                    # Re-ranker 효과 예상치 계산
                    if is_llm_generated:
                        st.success("🎯 **LLM 생성**: Re-ranker 효과 정확하게 측정 가능!")
                    else:
                        complex_count = sum(difficulty_counts.get('advanced', 0) for _ in range(1))
                        if len(benchmark_data) > 0:
                            complexity_ratio = complex_count / len(benchmark_data) * 100
                            st.info(f"🎯 Re-ranker 효과 예상: {complexity_ratio:.1f}% (복잡한 질문 비율)")
                        
                        # 질문 난이도별 분포 미니 차트
                        if len(difficulty_counts) > 1:
                            diff_df = pd.DataFrame(list(difficulty_counts.items()), 
                                                 columns=['Difficulty', 'Count'])
                            fig_mini = px.bar(diff_df, x='Difficulty', y='Count', 
                                            height=200, title='난이도 분포')
                            st.plotly_chart(fig_mini, use_container_width=True)
                        
            else:
                if benchmark_file_path:
                    st.error(f"벤치마크 파일 '{benchmark_file_path}'을 찾을 수 없습니다.")
                else:
                    st.warning("벤치마크 파일을 선택하거나 경로를 입력해주세요.")
                if not llm_files:
                    st.info("'LLM 정확한 벤치마크' 탭에서 먼저 정확한 데이터셋을 생성하세요.")
                benchmark_data = None
        
        if st.button("🚀 벤치마크 실행!", use_container_width=True, type="primary") and benchmark_data:
            progress_bar = st.progress(0, text="벤치마크 시작...")
            results_placeholder = st.empty()
            benchmark_results = []
            detailed_metrics_results = {}
            
            reranker = load_reranker_model() if use_reranker else None
            total_models = len(available_models)
            
            for i, (model_name, collection_name) in enumerate(available_models.items()):
                progress_bar.progress(i / total_models, text=f"[{i+1}/{total_models}] {model_name.split('/')[-1]}")
                
                try:
                    retriever_model = load_embedding_model(model_name)
                    collection = chroma_client.get_collection(name=collection_name)
                    
                    hit_count = 0
                    total_mrr = 0
                    total_ndcg = 0
                    total_precision = 0
                    
                    if use_batch:
                        queries = [item['query'] for item in benchmark_data]
                        n_results = initial_k if use_reranker else k_value
                        query_embeddings = retriever_model.encode(queries, batch_size=32, show_progress_bar=False)
                        
                        for idx, (item, embedding) in enumerate(zip(benchmark_data, query_embeddings)):
                            results = collection.query(
                                query_embeddings=[embedding.tolist()],
                                n_results=n_results,
                                include=["documents"]
                            )
                            
                            retrieved_ids = results['ids'][0]
                            
                            if use_reranker and reranker:
                                retrieved_docs = results['documents'][0]
                                pairs = [[item['query'], doc] for doc in retrieved_docs]
                                scores = reranker.predict(pairs)
                                reranked = sorted(zip(scores, retrieved_ids), key=lambda x: x[0], reverse=True)
                                final_ids = [doc_id for _, doc_id in reranked[:k_value]]
                            else:
                                final_ids = retrieved_ids[:k_value]
                            
                            relevant_docs = set(item['relevant_docs'])
                            is_hit = any(
                                r_id.startswith(ans_id + "_") 
                                for r_id in final_ids 
                                for ans_id in relevant_docs
                            )
                            
                            if is_hit:
                                hit_count += 1
                            
                            if calculate_detailed:
                                metrics = calculate_detailed_metrics(final_ids, relevant_docs, k_value)
                                total_mrr += metrics['mrr']
                                total_ndcg += metrics['ndcg']
                                total_precision += metrics['precision_at_k']
                    else:
                        for item in benchmark_data:
                            query = item['query']
                            relevant_docs = set(item['relevant_docs'])
                            
                            n_results = initial_k if use_reranker else k_value
                            query_embedding = retriever_model.encode(query).tolist()
                            results = collection.query(
                                query_embeddings=[query_embedding],
                                n_results=n_results,
                                include=["documents"]
                            )
                            
                            retrieved_ids = results['ids'][0]
                            
                            if use_reranker and reranker:
                                retrieved_docs = results['documents'][0]
                                pairs = [[query, doc] for doc in retrieved_docs]
                                scores = reranker.predict(pairs)
                                reranked = sorted(zip(scores, retrieved_ids), key=lambda x: x[0], reverse=True)
                                final_ids = [doc_id for _, doc_id in reranked[:k_value]]
                            else:
                                final_ids = retrieved_ids[:k_value]
                            
                            is_hit = any(
                                r_id.startswith(ans_id + "_") 
                                for r_id in final_ids 
                                for ans_id in relevant_docs
                            )
                            
                            if is_hit:
                                hit_count += 1
                            
                            if calculate_detailed:
                                metrics = calculate_detailed_metrics(final_ids, relevant_docs, k_value)
                                total_mrr += metrics['mrr']
                                total_ndcg += metrics['ndcg']
                                total_precision += metrics['precision_at_k']
                    
                    hit_rate = hit_count / len(benchmark_data)
                    avg_mrr = total_mrr / len(benchmark_data) if calculate_detailed else 0
                    avg_ndcg = total_ndcg / len(benchmark_data) if calculate_detailed else 0
                    avg_precision = total_precision / len(benchmark_data) if calculate_detailed else 0
                    
                    result_entry = {
                        "Model": model_name,
                        "Re-ranked": "Yes" if use_reranker else "No", 
                        "Batch": "Yes" if use_batch else "No",
                        f"Hit Rate @{k_value}": f"{hit_rate:.2%}"
                    }
                    
                    if calculate_detailed:
                        result_entry.update({
                            "MRR": f"{avg_mrr:.3f}",
                            "NDCG": f"{avg_ndcg:.3f}",
                            "Precision@K": f"{avg_precision:.3f}"
                        })
                        detailed_metrics_results[model_name] = [hit_rate, avg_mrr, avg_ndcg, avg_precision]
                    
                    benchmark_results.append(result_entry)
                    
                except Exception as e:
                    error_entry = {
                        "Model": model_name,
                        "Re-ranked": "Yes" if use_reranker else "No",
                        "Batch": "Yes" if use_batch else "No", 
                        f"Hit Rate @{k_value}": f"오류: {str(e)[:50]}"
                    }
                    benchmark_results.append(error_entry)
                
                results_placeholder.dataframe(pd.DataFrame(benchmark_results), use_container_width=True)
            
            progress_bar.progress(1.0, text="✅ 벤치마크 완료!")
            
            # 세션 상태에 결과 저장 (그래프 탭에서 사용)
            st.session_state['benchmark_results'] = benchmark_results
            st.session_state['detailed_metrics_results'] = detailed_metrics_results
            
            # 간단한 시각화
            if len(benchmark_results) > 1:
                st.subheader("📊 결과 시각화")
                try:
                    fig = create_model_comparison_chart(benchmark_results)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True, key="model_comparison_chart")
                except Exception as e:
                    st.error(f"차트 생성 실패: {e}")
            
            if st.button("📁 결과 저장"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"benchmark_results_{timestamp}.csv"
                pd.DataFrame(benchmark_results).to_csv(filename, index=False, encoding='utf-8-sig')
                st.success(f"결과가 '{filename}'로 저장되었습니다!")

        # 디버깅 및 문제 해결 섹션
        st.markdown("---")
        st.subheader("🔧 디버깅 및 문제 해결")
        
        with st.expander("🚨 성능이 낮거나 오류가 발생할 때 사용하세요"):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🏥 컬렉션 상태 체크"):
                    health_df = check_collection_health()
                    if health_df is not None:
                        st.session_state['collections_health'] = health_df
                
                if st.button("🧪 간단 성능 테스트"):
                    # 간단한 키워드 검색 테스트
                    simple_queries = ["모기", "제품", "효과", "사용", "추천"]
                    
                    try:
                        test_results = []
                        for model_name, collection_name in list(available_models.items())[:3]:
                            retriever_model = load_embedding_model(model_name)
                            collection = chroma_client.get_collection(collection_name)
                            
                            hits = 0
                            for query in simple_queries:
                                query_embedding = retriever_model.encode(query).tolist()
                                results = collection.query(query_embeddings=[query_embedding], n_results=5)
                                
                                # 단순 키워드 포함 여부 확인
                                docs = results.get('documents', [[]])[0]
                                if any(query in str(doc) for doc in docs):
                                    hits += 1
                            
                            hit_rate = hits / len(simple_queries) * 100
                            test_results.append({
                                'Model': model_name.split('/')[-1],
                                'Simple Hit Rate': f"{hit_rate:.1f}%"
                            })
                        
                        st.dataframe(pd.DataFrame(test_results))
                        
                        # 평균 성능 분석
                        rates = [float(r['Simple Hit Rate'].replace('%', '')) for r in test_results if '%' in r['Simple Hit Rate']]
                        if rates:
                            avg = sum(rates) / len(rates)
                            if avg > 70:
                                st.success(f"✅ 기본 검색 기능 정상 (평균 {avg:.1f}%)")
                            elif avg > 30:
                                st.warning(f"⚠️ 기본 검색 기능 보통 (평균 {avg:.1f}%)")
                            else:
                                st.error(f"❌ 기본 검색 기능 불량 (평균 {avg:.1f}%) - 컬렉션 재구축 필요")
                        
                    except Exception as e:
                        st.error(f"간단 테스트 실패: {e}")
            
            with col2:
                if st.button("🔍 벤치마크 데이터 검증"):
                    # LLM 파일 먼저 확인
                    llm_files = [f for f in os.listdir('.') if f.startswith('llm_benchmark_') and f.endswith('.json')]
                    
                    if llm_files:
                        latest_llm_file = max(llm_files, key=lambda x: os.path.getmtime(x))
                        with open(latest_llm_file, 'r', encoding='utf-8') as f:
                            benchmark_data = json.load(f)
                        
                        st.write(f"📊 LLM 벤치마크 파일: {latest_llm_file}")
                        st.write(f"📊 총 질문 수: {len(benchmark_data)}")
                        
                        # LLM 생성 여부 확인
                        is_llm = any(item.get('generation_method') == 'document_based' for item in benchmark_data)
                        if is_llm:
                            st.success("✅ LLM 기반 정확한 벤치마크입니다!")
                        
                        # 참조 문서 ID 분석
                        all_ref_ids = set()
                        for item in benchmark_data:
                            all_ref_ids.update(item.get('relevant_docs', []))
                        
                        st.write(f"📋 참조 문서 ID 수: {len(all_ref_ids)}")
                        st.write(f"📝 샘플 ID: {list(all_ref_ids)[:5]}")
                        
                        # 컬렉션과의 일치도 확인
                        if available_models:
                            first_collection_name = list(available_models.values())[0]
                            collection = chroma_client.get_collection(first_collection_name)
                            
                            # 샘플 검증
                            sample_ids = list(all_ref_ids)[:5]
                            found_count = 0
                            
                            try:
                                all_collection_data = collection.get(include=['ids'])
                                collection_ids = all_collection_data.get('ids', [])
                                
                                for ref_id in sample_ids:
                                    matching_chunks = [cid for cid in collection_ids if cid.startswith(ref_id + "_")]
                                    if matching_chunks:
                                        found_count += 1
                                
                                match_ratio = found_count / len(sample_ids) * 100
                                
                                if match_ratio > 80:
                                    st.success(f"✅ 데이터 일치도 우수 ({match_ratio:.1f}%)")
                                elif match_ratio > 40:
                                    st.warning(f"⚠️ 데이터 일치도 보통 ({match_ratio:.1f}%)")
                                else:
                                    st.error(f"❌ 데이터 불일치 심각 ({match_ratio:.1f}%) - 컬렉션 재구축 필요")
                                    
                            except Exception as e:
                                st.error(f"일치도 확인 실패: {e}")
                    else:
                        st.error("LLM 벤치마크 파일이 없습니다. 먼저 생성해주세요.")
                
                # 개별 모델 디버깅
                if available_models:
                    selected_debug_model = st.selectbox("디버깅할 모델:", list(available_models.keys()), key="debug_model")
                    
                    if st.button("🔍 선택 모델 상세 디버깅"):
                        # LLM 파일 찾기
                        llm_files = [f for f in os.listdir('.') if f.startswith('llm_benchmark_') and f.endswith('.json')]
                        
                        if llm_files:
                            latest_llm_file = max(llm_files, key=lambda x: os.path.getmtime(x))
                            with open(latest_llm_file, 'r', encoding='utf-8') as f:
                                benchmark_data = json.load(f)
                            
                            debug_benchmark_performance(
                                selected_debug_model,
                                available_models[selected_debug_model],
                                benchmark_data[:3]  # 처음 3개만 디버깅
                            )
                        else:
                            st.error("LLM 벤치마크 파일이 없습니다.")
            
            # 문제 해결 가이드
            st.markdown("---")
            st.subheader("💡 문제 해결 가이드")
            
            st.markdown("""
            **🔥 성능이 매우 낮은 경우 (10% 미만)**:
            1. **LLM 벤치마크 사용**: 탭2에서 정확한 벤치마크 데이터셋 생성
            2. **컬렉션 재구축**: 탭3에서 청크 크기를 늘려서 재구축 (500→1000)
            3. **데이터 일치도 확인**: 위의 "벤치마크 데이터 검증" 실행
            
            **⚠️ 일부 모델에서 오류가 발생하는 경우**:
            1. **GPU 메모리 부족**: 배치 처리 끄기 또는 모델 하나씩 테스트
            2. **모델 로딩 실패**: 인터넷 연결 확인 또는 모델명 확인
            3. **컬렉션 누락**: "컬렉션 상태 체크"로 확인
            
            **📊 Re-ranker 효과가 없는 경우**:
            1. **LLM 벤치마크 사용**: 정확한 질문으로 Re-ranker 효과 제대로 측정
            2. **K값이 너무 작음**: 1차 검색을 50개 이상으로 늘리기
            3. **기본 성능이 이미 좋음**: 정상적인 현상일 수 있음
            """)

# --- TAB 5: 성능 분석 ---
with tab5:
    st.header("📈 상세 성능 분석 및 시각화")
    
    if 'benchmark_results' not in st.session_state:
        st.warning("먼저 '벤치마크 실행' 탭에서 벤치마크를 실행해주세요.")
    else:
        benchmark_results = st.session_state['benchmark_results']
        detailed_metrics = st.session_state.get('detailed_metrics_results', {})
        
        # 분석 유형 선택
        analysis_type = st.selectbox(
            "분석 유형을 선택하세요:",
            ["모델 성능 비교", "Re-ranker 효과 분석", "K값 변화 분석", "질문 유형별 분석", "종합 대시보드"]
        )
        
        if analysis_type == "모델 성능 비교":
            st.subheader("🏆 모델별 성능 순위")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 막대 차트
                fig_bar = create_model_comparison_chart(benchmark_results)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                # 상세 메트릭 레이더 차트 (있는 경우)
                if detailed_metrics:
                    fig_radar = create_detailed_metrics_radar_chart(detailed_metrics)
                    st.plotly_chart(fig_radar, use_container_width=True)
                else:
                    st.info("상세 메트릭을 보려면 벤치마크 실행 시 '상세 메트릭 계산'을 체크하세요.")
            
            # 성능 순위 테이블
            df_results = pd.DataFrame(benchmark_results)
            hit_rate_col = [col for col in df_results.columns if "Hit Rate" in col][0]
            df_results['hit_rate_numeric'] = df_results[hit_rate_col].str.replace('%', '').str.replace('오류.*', '0', regex=True).astype(float)
            df_sorted = df_results.sort_values('hit_rate_numeric', ascending=False)
            
            st.subheader("📋 성능 순위표")
            st.dataframe(df_sorted[['Model', hit_rate_col, 'Re-ranked', 'Batch']], use_container_width=True)
            
        elif analysis_type == "Re-ranker 효과 분석":
            st.subheader("🔄 Re-ranker 효과 분석")
            
            # Re-ranker 사용/미사용 결과가 있는지 확인
            reranker_yes = [r for r in benchmark_results if r.get('Re-ranked') == 'Yes']
            reranker_no = [r for r in benchmark_results if r.get('Re-ranked') == 'No']
            
            if reranker_yes and reranker_no:
                # 비교 차트 생성
                comparison_data = []
                for model in set([r['Model'] for r in benchmark_results]):
                    yes_result = next((r for r in reranker_yes if r['Model'] == model), None)
                    no_result = next((r for r in reranker_no if r['Model'] == model), None)
                    
                    if yes_result and no_result:
                        hit_rate_col = [col for col in yes_result.keys() if "Hit Rate" in col][0]
                        comparison_data.append({
                            'Model': model.split('/')[-1],
                            'Without_Reranker': float(no_result[hit_rate_col].replace('%', '')),
                            'With_Reranker': float(yes_result[hit_rate_col].replace('%', ''))
                        })
                
                if comparison_data:
                    fig_comparison = create_reranker_comparison_chart(comparison_data)
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # 개선 효과 계산
                    improvements = []
                    for data in comparison_data:
                        improvement = data['With_Reranker'] - data['Without_Reranker']
                        improvements.append({
                            'Model': data['Model'],
                            'Improvement': f"{improvement:+.2f}%"
                        })
                    
                    st.subheader("📊 Re-ranker 개선 효과")
                    st.dataframe(pd.DataFrame(improvements), use_container_width=True)
                else:
                    st.info("Re-ranker 비교 데이터가 충분하지 않습니다.")
            else:
                st.info("Re-ranker 사용/미사용 비교를 위해서는 두 가지 설정으로 벤치마크를 실행해야 합니다.")
                
        elif analysis_type == "K값 변화 분석":
            st.subheader("📈 K값에 따른 성능 변화 시뮬레이션")
            
            st.info("이 분석을 위해서는 다양한 K값으로 벤치마크를 실행해야 합니다.")
            
            # LLM 벤치마크 기반 시뮬레이션 데이터
            sample_k_analysis = {}
            for model in [r['Model'] for r in benchmark_results[:3]]:
                k_values = list(range(1, 21))
                # LLM 벤치마크에서는 더 높은 기본 성능을 시뮬레이션
                base_rate = 0.6  # LLM 기반이므로 더 높은 시작점
                hit_rates = [min(0.95, base_rate + (k-1) * 0.02 + random.uniform(-0.03, 0.03)) for k in k_values]
                sample_k_analysis[model] = {'k_values': k_values, 'hit_rates': [h*100 for h in hit_rates]}
            
            fig_k_analysis = create_k_value_analysis_chart(sample_k_analysis)
            st.plotly_chart(fig_k_analysis, use_container_width=True)
            
            st.info("💡 LLM 벤치마크를 사용하면 더 정확한 K값 최적화가 가능합니다.")
                
        elif analysis_type == "질문 유형별 분석":
            st.subheader("🎯 질문 유형별 성능 분석")
            
            # LLM 벤치마크 파일 우선 확인
            llm_files = [f for f in os.listdir('.') if f.startswith('llm_benchmark_') and f.endswith('.json')]
            
            if llm_files:
                latest_llm_file = max(llm_files, key=lambda x: os.path.getmtime(x))
                
                with open(latest_llm_file, 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)
                
                st.success(f"✅ LLM 벤치마크 분석: {latest_llm_file} ({len(benchmark_data)}개 질문)")
                
                # 질문 유형별 분포 및 복잡도 분석
                type_counts = {}
                difficulty_counts = {}
                complexity_analysis = {}
                
                for item in benchmark_data:
                    q_type = item.get('question_type', 'unknown')
                    difficulty = item.get('difficulty', 'basic')
                    relevant_docs_count = len(item.get('relevant_docs', []))
                    
                    type_counts[q_type] = type_counts.get(q_type, 0) + 1
                    difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
                    
                    if q_type not in complexity_analysis:
                        complexity_analysis[q_type] = {'total': 0, 'multi_doc': 0, 'advanced': 0}
                    
                    complexity_analysis[q_type]['total'] += 1
                    if relevant_docs_count > 1:
                        complexity_analysis[q_type]['multi_doc'] += 1
                    if difficulty == 'advanced':
                        complexity_analysis[q_type]['advanced'] += 1
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # 질문 유형 분포 파이 차트
                    type_df = pd.DataFrame(list(type_counts.items()), columns=['Question Type', 'Count'])
                    fig_pie = px.pie(type_df, values='Count', names='Question Type', 
                                   title='질문 유형 분포')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # 난이도 분포 막대 차트
                    difficulty_df = pd.DataFrame(list(difficulty_counts.items()), 
                                               columns=['Difficulty', 'Count'])
                    fig_bar = px.bar(difficulty_df, x='Difficulty', y='Count', 
                                   title='난이도 분포', color='Count',
                                   color_discrete_sequence=['lightcoral', 'lightblue'])
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col3:
                    # 복잡도 분석 차트
                    complexity_data = []
                    for q_type, data in complexity_analysis.items():
                        complexity_score = (data['multi_doc'] + data['advanced']) / data['total'] * 100
                        complexity_data.append({
                            'Question Type': q_type,
                            'Complexity Score': complexity_score
                        })
                    
                    complexity_df = pd.DataFrame(complexity_data)
                    fig_complexity = px.bar(complexity_df, x='Question Type', y='Complexity Score',
                                          title='질문 유형별 복잡도', 
                                          color='Complexity Score',
                                          color_continuous_scale='Reds')
                    fig_complexity.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_complexity, use_container_width=True)
                
                # 상세 분석 테이블
                st.subheader("📋 질문 유형별 상세 분석")
                
                analysis_table = []
                for q_type, data in complexity_analysis.items():
                    total = data['total']
                    multi_doc_ratio = data['multi_doc'] / total * 100 if total > 0 else 0
                    advanced_ratio = data['advanced'] / total * 100 if total > 0 else 0
                    complexity_score = (data['multi_doc'] + data['advanced']) / total * 100 if total > 0 else 0
                    
                    # Re-ranker 효과 예상
                    if complexity_score > 50:
                        reranker_effect = "높음 🔥"
                    elif complexity_score > 25:
                        reranker_effect = "중간 ⚡"
                    else:
                        reranker_effect = "낮음 📝"
                    
                    analysis_table.append({
                        '질문 유형': q_type,
                        '개수': total,
                        '비율': f"{total/len(benchmark_data)*100:.1f}%",
                        '멀티 문서 비율': f"{multi_doc_ratio:.1f}%",
                        '고급 난이도 비율': f"{advanced_ratio:.1f}%",
                        '복잡도 점수': f"{complexity_score:.1f}%",
                        'Re-ranker 효과 예상': reranker_effect
                    })
                
                st.dataframe(pd.DataFrame(analysis_table), use_container_width=True)
                
                # 인사이트 제공
                st.subheader("💡 분석 인사이트")
                
                high_complexity_types = [item for item in analysis_table if float(item['복잡도 점수'].replace('%', '')) > 50]
                if high_complexity_types:
                    st.success(f"🎯 **Re-ranker 효과가 클 것으로 예상되는 질문 유형**: {', '.join([item['질문 유형'] for item in high_complexity_types])}")
                
                total_advanced = sum(data['advanced'] for data in complexity_analysis.values())
                advanced_ratio = total_advanced / len(benchmark_data) * 100
                
                if advanced_ratio > 60:
                    st.info(f"🔥 **고급 질문 비율이 높음** ({advanced_ratio:.1f}%): Re-ranker 성능 향상이 크게 나타날 것으로 예상됩니다.")
                elif advanced_ratio > 30:
                    st.info(f"⚡ **중간 수준의 고급 질문** ({advanced_ratio:.1f}%): Re-ranker 효과가 적당히 나타날 것으로 예상됩니다.")
                else:
                    st.warning(f"📝 **기본 질문 위주** ({advanced_ratio:.1f}%): Re-ranker 효과가 제한적일 수 있습니다. 더 복잡한 질문 생성을 고려해보세요.")
            else:
                st.warning("LLM 벤치마크 파일이 없습니다. 먼저 'LLM 정확한 벤치마크' 탭에서 생성해주세요.")
                
        elif analysis_type == "종합 대시보드":
            st.subheader("🎛️ 종합 성능 대시보드")
            
            # 메트릭 요약
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_hit_rate = np.mean([
                    float(r[next(col for col in r.keys() if "Hit Rate" in col)].replace('%', '')) 
                    for r in benchmark_results if '%' in str(r.get(next(col for col in r.keys() if "Hit Rate" in col), ''))
                ])
                st.metric("평균 Hit Rate", f"{avg_hit_rate:.1f}%")
            
            with col2:
                total_models = len([r for r in benchmark_results if 'Model' in r])
                st.metric("테스트된 모델 수", total_models)
            
            with col3:
                reranker_models = len([r for r in benchmark_results if r.get('Re-ranked') == 'Yes'])
                st.metric("Re-ranker 사용 모델", reranker_models)
            
            with col4:
                gpu_models = len([r for r in benchmark_results if r.get('Batch') == 'Yes'])
                st.metric("GPU 배치 처리", gpu_models)
            
            # 성능 히트맵
            if detailed_metrics:
                st.subheader("🔥 성능 히트맵")
                
                models = list(detailed_metrics.keys())
                metrics_names = ['Hit Rate', 'MRR', 'NDCG', 'Precision@K']
                performance_matrix = np.array([detailed_metrics[model] for model in models])
                
                fig_heatmap = px.imshow(
                    performance_matrix,
                    x=metrics_names,
                    y=[model.split('/')[-1] for model in models],
                    color_continuous_scale='RdYlGn',
                    title='모델 × 메트릭 성능 히트맵',
                    text_auto='.3f'
                )
                fig_heatmap.update_layout(height=400)
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # 전체 결과 테이블
            st.subheader("📋 전체 벤치마크 결과")
            st.dataframe(pd.DataFrame(benchmark_results), use_container_width=True)

# --- 사이드바 ---
with st.sidebar:
    st.header("📊 시스템 정보")
    
    if 'json_data' in st.session_state:
        items = st.session_state['json_data'].get('items', st.session_state['json_data'])
        st.success(f"✅ 데이터 로드됨: {len(items)}개 문서")
    else:
        st.info("📁 데이터를 로드해주세요")
    
    # 벤치마크 파일 상태
    # LLM 생성 파일 확인
    try:
        llm_files = [f for f in os.listdir('.') if f.startswith('llm_benchmark_') and f.endswith('.json')]
        
        if llm_files:
            st.success(f"✅ LLM 벤치마크: {len(llm_files)}개 파일")
            latest_llm_file = max(llm_files, key=lambda x: os.path.getmtime(x))
            
            with open(latest_llm_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)
            st.write(f"📝 최신: {latest_llm_file} ({len(llm_data)}개 질문)")
        else:
            st.info("📝 LLM 벤치마크를 생성해주세요")
    except:
        st.info("📝 LLM 벤치마크를 생성해주세요")
    
    # 컬렉션 상태
    try:
        chroma_client = get_chroma_client()
        collections = chroma_client.list_collections()
        if collections:
            st.success(f"✅ 컬렉션: {len(collections)}개 구축됨")
            for col in collections:
                collection_obj = chroma_client.get_collection(col.name)
                st.write(f"- {col.name} ({collection_obj.count()}개)")
        else:
            st.info("🔧 컬렉션을 구축해주세요")
    except:
        st.warning("❌ ChromaDB 연결 실패")
    
    st.markdown("---")
    
    # 빠른 액션 버튼들
    st.subheader("⚡ 빠른 액션")
    
    if st.button("🔄 전체 초기화", help="모든 세션 상태를 초기화합니다"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("초기화 완료!")
        st.rerun()
    
    if st.button("📱 시스템 상태 체크"):
        st.write("**GPU 상태:**", "✅ 사용 가능" if device == "cuda" else "❌ 사용 불가")
        
        try:
            torch.cuda.empty_cache()
            if device == "cuda":
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                st.write(f"**GPU 메모리:** {memory_allocated:.1f}GB / {memory_reserved:.1f}GB")
        except:
            pass
        
        st.write(f"**ChromaDB 경로:** {CHROMA_DB_PATH}")
    
    # 고급 설정
    with st.expander("🔧 고급 설정"):
        debug_mode = st.checkbox("디버그 모드")
        if debug_mode:
            st.write("**세션 상태:**")
            for key in st.session_state.keys():
                if key == 'json_data':
                    st.write(f"- {key}: 로드됨")
                elif key == 'benchmark_results':
                    st.write(f"- {key}: {len(st.session_state[key])}개 결과")
                else:
                    st.write(f"- {key}: {type(st.session_state[key])}")
        
        auto_save = st.checkbox("자동 저장", value=True)
        show_warnings = st.checkbox("경고 표시", value=True)
        
        if st.button("🧹 캐시 정리"):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.success("캐시 정리 완료!")

st.markdown("---")
st.markdown("**💡 사용 팁:** 탭을 순서대로 진행하세요: 📁 데이터 준비 → **🤖 LLM 정확한 벤치마크** → 🔧 컬렉션 구축 → 📊 벤치마크 실행 → 📈 성능 분석")
st.markdown("**🎯 권장:** LLM 방식으로 정확한 벤치마크를 생성하면 신뢰할 수 있는 성능 측정이 가능합니다!")
