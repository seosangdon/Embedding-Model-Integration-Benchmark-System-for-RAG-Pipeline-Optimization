### *Embedding Model Integration Benchmark System for RAG Pipeline Optimization*
### *** Performance benchmark test *** ###
#---
### **Ⅰ. 개요**

> AI 기반 챗봇 어시스턴트의 답변 품질에 핵심적인 영향을 미치는 RAG(검색 증강 생성) 파이프라인의 성능을 극대화하는 것을 목표로 합니다. 이를 위해, 독립적인 **'통합 벤치마크 시스템'**을 직접 개발하여 다양한 임베딩 모델(Retriever)과 재순위 모델(Re-ranker)의 조합을 실제 데이터 환경에서 체계적으로 평가하고, 최적의 아키텍처를 도출했습니다.
> 
- **주요 목표**:
    1. 다양한 임베딩 모델의 검색 성능을 객관적인 지표로 비교 분석.
    2. 청킹(Chunking) 및 재순위(Re-ranking) 등 고급 RAG 기법의 효과를 데이터로 검증.
    3. 최종적으로, 실제 서비스에 적용할 최고 성능의 모델 조합을 데이터에 기반하여 선정.
- **관련 프로젝트**: [AI 날씨 어시스턴트](https://www.google.com/search?q=https://github.com/your-repo/your-weather-assistant-project) (가상 링크)

### **Ⅱ. 벤치마크 설계 (Benchmark Design)**

**A. 평가 데이터셋 (Test Dataset)**

- **원본 데이터**: 네이버 API를 통해 **'모기용품'**이라는 특정 키워드로 수집된 **실제 한국어 블로그 포스트 100건**을 원본 데이터로 사용했습니다.
- **평가 데이터셋 구축**: 단순히 공개된 데이터셋을 사용하는 것을 넘어, **LLM(GPT-4o-mini)을 활용한 '질문 자동 생성 파이프라인'**을 직접 구축했습니다. 이 시스템은 수집된 문서의 실제 내용을 기반으로, '사실 확인', '비교', '추론' 등 다양한 유형의 **현실적인 질문-정답 쌍 40개**를 생성하여 평가의 신뢰도와 타당성을 확보했습니다. 사용자는 이 시스템을 통해 처리할 문서 수와 문서당 질문 수를 직접 조절하여, 테스트 케이스의 양과 질을 유연하게 관리할 수 있습니다.

**B. 평가 지표 (Metrics)**

RAG 파이프라인의 검색 성능을 다각적으로 평가하기 위해 다음과 같은 표준 정보 검색(IR) 지표를 사용했습니다.

- **Hit Rate @K**: 상위 K개의 검색 결과에 정답이 포함될 확률. (주요 지표)
- **MRR (Mean Reciprocal Rank)**: 정답을 얼마나 높은 순위에서 찾아냈는지 평가. (1위에 가까울수록 좋음)
- **NDCG @K (Normalized Discounted Cumulative Gain)**: 검색 결과 상위권의 순서와 품질을 종합적으로 평가.

**C. 평가 대상 모델**

한국어 및 다국어 환경에서 널리 사용되는 최신 임베딩 모델 **총 8종**을 대상으로 성능을 비교했습니다.

- **Retriever (1차 검색 모델)**
    - **한국어 특화 그룹**: `jhgan/ko-sroberta-multitask`, `BM-K/KoSimCSE-roberta-multitask`, `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
    - **다국어/검색 특화 그룹**: `intfloat/multilingual-e5-base`, `intfloat/multilingual-e5-large`
    - **범용 다국어 그룹**: `distiluse-base-multilingual-cased-v1`, `paraphrase-multilingual-MiniLM-L12-v2`
    - **대조군 (영어 모델)**: `BAAI/bge-large-en-v1.5`
- **Re-ranker (2차 검토 모델)**
    - `BAAI/bge-reranker-v2-m3` (글로벌 SOTA Re-ranker)

### **Ⅲ. 성능 최적화 실험 및 검증**

구축된 벤치마크 시스템을 활용하여, 다음과 같은 체계적인 성능 최적화 실험을 진행했습니다.

1. **청킹(Chunking) 전략 도입**: 긴 문서를 500자 단위의 의미 있는 '조각'으로 분할 저장하여, 검색의 정확도를 높이는 기본 튜닝을 진행했습니다.
2. **재순위 매기기(Re-ranking) 도입**: 1차 검색(Retriever)으로 가져온 후보군을, 2차로 더 정교한 모델(Re-ranker)이 재검토하여 최종 순위를 매기는 2-stage 파이프라인을 도입했습니다.
3. **모델 조합 비교 분석**: 총 8개의 Retriever와 Re-ranker 적용 여부에 따른 성능 변화를 데이터로 검증했습니다.

### **4. 검증 결과 및 최종 결론**

**[벤치마크 최종 결과 요약 (K=10)]**

| Retriever 모델 (1차 검색) | Re-ranker 미사용 (Hit Rate) | Re-ranker 사용 (+ `bge-reranker`) | 성능 향상폭 |
| --- | --- | --- | --- |
| **`intfloat/multilingual-e5-base`** | **95.00%** | **100.00%** | **+5.00%p** |
| `jhgan/ko-sroberta-multitask` | 87.50% | 97.50% | +10.00%p |
| `intfloat/multilingual-e5-large` | 90.00% | 97.50% | +7.50%p |
| `BM-K/KoSimCSE-roberta-multitask` | 77.50% | 97.50% | **+20.00%p** |

Sheets로 내보내기

실험 결과, 1차 검색에서는 검색에 특화된 다국어 모델인 `intfloat/multilingual-e5-base`가 95%로 가장 높은 성능을 보였고, **Re-ranker를 적용했을 때 Hit Rate 100%라는 완벽한 성능을 달성**하는 것을 확인했습니다. 특히 `KoSimCSE` 모델은 Re-ranker 적용 후 성능이 **20%p**나 극적으로 상승하며, 2-Stage 파이프라인 전략의 유효성을 명확히 증명했습니다.

이를 통해, 단순히 단일 모델의 성능에 의존하는 것을 넘어, **`Retriever + Re-ranker` 파이프라인이 RAG 시스템의 성능을 극대화하는 최적의 아키텍처임을 데이터 기반으로 입증**하고 최종 채택했습니다. 이 과정은 AI 답변의 신뢰성과 정확성을 확보하는 핵심적인 기반이 되었습니다.
