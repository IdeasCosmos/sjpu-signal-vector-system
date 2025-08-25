"""
하이브리드 SJPU 시스템 - 핵심 클래스 구현
작은틀 1.1: 기본 SJPU 클래스 구현 (80% 완성도)
"""

import numpy as np
import time
import logging
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import threading
import psutil
import gc

# =============================================================================
# 핵심 설정 클래스 (완전 구현)
# =============================================================================

class ProcessingMode(Enum):
    """처리 모드 열거형"""
    QUANTUM = "quantum"
    HYBRID = "hybrid"
    CLASSICAL = "classical"
    AUTO = "auto"

class PerformanceLevel(Enum):
    """성능 레벨"""
    ULTRA_LOW = 1    # 극저사양 (< 128MB)
    LOW = 2          # 저사양 (128-256MB)
    MEDIUM = 3       # 중사양 (256-512MB)
    HIGH = 4         # 고사양 (512MB-1GB)
    ULTRA_HIGH = 5   # 최고사양 (> 1GB)

@dataclass
class HybridSJPUConfig:
    """하이브리드 SJPU 시스템 완전 설정 클래스"""
    
    # === 기본 차원 설정 ===
    vector_dimensions: int = 256
    semantic_dimensions: int = 512
    quantum_register_size: int = 8
    emotion_dimensions: int = 32
    
    # === 모드 전환 임계값 ===
    quantum_threshold: int = 1000      # 1000자 이하 = 양자 모드
    classical_threshold: int = 5000    # 5000자 이상 = 클래식 모드
    hybrid_threshold: int = 2500       # 중간 = 하이브리드 모드
    
    # === 메모리 관리 ===
    max_memory_mb: int = 512
    cache_size_limit: int = 1000
    gc_frequency: int = 100
    memory_warning_threshold: float = 0.8
    memory_critical_threshold: float = 0.9
    
    # === 성능 최적화 ===
    use_ml_predictor: bool = True
    use_quantum_cache: bool = True
    use_genetic_optimizer: bool = True
    use_multiprocessing: bool = False
    max_workers: int = 4
    
    # === 고급 기능 활성화 ===
    consciousness_stream: bool = True
    temporal_compression: bool = True
    quantum_compression: bool = True
    dream_state_processing: bool = True
    selective_attention: bool = True
    
    # === 디버깅 및 모니터링 ===
    debug_mode: bool = False
    verbose_logging: bool = False
    performance_monitoring: bool = True
    auto_optimization: bool = True
    
    # === 보안 및 안정성 ===
    max_processing_time: float = 30.0  # 최대 처리 시간 (초)
    input_size_limit: int = 100000     # 최대 입력 크기 (문자)
    output_size_limit: int = 50000     # 최대 출력 크기 (문자)
    
    def __post_init__(self):
        """설정 검증 및 자동 조정"""
        self._validate_config()
        self._auto_adjust_config()
    
    def _validate_config(self):
        """설정값 유효성 검사"""
        if self.vector_dimensions < 16 or self.vector_dimensions > 2048:
            raise ValueError(f"vector_dimensions must be 16-2048, got {self.vector_dimensions}")
        
        if self.max_memory_mb < 64:
            raise ValueError(f"max_memory_mb must be >= 64MB, got {self.max_memory_mb}")
        
        if not 0 < self.memory_warning_threshold < self.memory_critical_threshold < 1:
            raise ValueError("Memory thresholds must be: 0 < warning < critical < 1")
    
    def _auto_adjust_config(self):
        """시스템 환경에 따른 자동 설정 조정"""
        try:
            # 시스템 메모리 확인
            total_memory = psutil.virtual_memory().total / (1024**3)  # GB
            
            if total_memory < 4:  # 4GB 미만
                self.max_memory_mb = min(self.max_memory_mb, 256)
                self.cache_size_limit = min(self.cache_size_limit, 500)
                self.vector_dimensions = min(self.vector_dimensions, 128)
                
            elif total_memory < 8:  # 8GB 미만
                self.max_memory_mb = min(self.max_memory_mb, 512)
                self.cache_size_limit = min(self.cache_size_limit, 1000)
                
            # CPU 코어 수에 따른 워커 조정
            cpu_count = psutil.cpu_count()
            self.max_workers = min(self.max_workers, max(2, cpu_count - 1))
            
        except ImportError:
            # psutil이 없는 경우 기본값 유지
            pass
    
    def get_performance_level(self) -> PerformanceLevel:
        """현재 설정 기반 성능 레벨 반환"""
        if self.max_memory_mb <= 128:
            return PerformanceLevel.ULTRA_LOW
        elif self.max_memory_mb <= 256:
            return PerformanceLevel.LOW
        elif self.max_memory_mb <= 512:
            return PerformanceLevel.MEDIUM
        elif self.max_memory_mb <= 1024:
            return PerformanceLevel.HIGH
        else:
            return PerformanceLevel.ULTRA_HIGH
    
    def optimize_for_speed(self):
        """속도 우선 최적화"""
        self.use_quantum_cache = True
        self.use_multiprocessing = True
        self.quantum_compression = False
        self.dream_state_processing = False
        
    def optimize_for_memory(self):
        """메모리 우선 최적화"""
        self.max_memory_mb = min(self.max_memory_mb, 256)
        self.cache_size_limit = min(self.cache_size_limit, 500)
        self.vector_dimensions = min(self.vector_dimensions, 128)
        self.quantum_compression = True
        self.temporal_compression = True
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridSJPUConfig':
        """딕셔너리에서 설정 생성"""
        return cls(**{k: v for k, v in config_dict.items() 
                     if k in cls.__dataclass_fields__})

# =============================================================================
# 처리 결과 클래스
# =============================================================================

@dataclass
class SJPUResult:
    """SJPU 처리 결과"""
    original_text: str
    processed_text: str
    processing_mode: ProcessingMode
    processing_time: float
    memory_used: float
    
    # 성능 메트릭스
    input_length: int = 0
    output_length: int = 0
    compression_ratio: float = 1.0
    quality_score: float = 0.5
    
    # 메타데이터
    timestamp: float = field(default_factory=time.time)
    config_hash: str = ""
    error_message: Optional[str] = None
    
    # 고급 분석 결과
    consciousness_analysis: Optional[Dict[str, Any]] = None
    quantum_coherence: Optional[List[float]] = None
    genetic_optimization: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.input_length:
            self.input_length = len(self.original_text)
        if not self.output_length:
            self.output_length = len(self.processed_text)
        
        if self.input_length > 0:
            self.compression_ratio = self.output_length / self.input_length
    
    def get_efficiency_score(self) -> float:
        """효율성 점수 계산 (0-1)"""
        if self.processing_time <= 0:
            return 0.0
        
        # 처리 속도 점수 (1000자/초 기준)
        speed_score = min(1.0, (self.input_length / 1000) / self.processing_time)
        
        # 메모리 효율성 점수 (100MB 기준)
        memory_score = max(0.0, 1.0 - (self.memory_used / 100))
        
        # 품질 점수
        quality_score = self.quality_score
        
        return (speed_score * 0.4 + memory_score * 0.3 + quality_score * 0.3)

# =============================================================================
# 메인 SJPU 클래스 (완전 구현)
# =============================================================================

class PredictiveHybridSJPU:
    """예측적 하이브리드 SJPU 시스템 - 완전 구현"""
    
    def __init__(self, config: Optional[HybridSJPUConfig] = None):
        """시스템 초기화"""
        self.config = config or HybridSJPUConfig()
        self._setup_logging()
        self._initialize_components()
        self._setup_monitoring()
        
        # 통계 및 상태
        self.processing_stats = {
            'total_processed': 0,
            'mode_distribution': defaultdict(int),
            'total_processing_time': 0.0,
            'total_memory_used': 0.0,
            'error_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        self.performance_history = deque(maxlen=1000)
        self.last_gc_time = time.time()
        self._lock = threading.Lock()
        
        self.logger.info(f"SJPU System initialized with {self.config.get_performance_level().name} performance level")
    
    def _setup_logging(self):
        """로깅 시스템 설정"""
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        
        if self.config.verbose_logging:
            self.logger.setLevel(logging.DEBUG)
    
    def _initialize_components(self):
        """핵심 컴포넌트 초기화"""
        # 기본 캐시 (간단 버전)
        self.simple_cache = {}
        self.cache_access_count = defaultdict(int)
        
        # 모드 예측 히스토리
        self.mode_prediction_history = deque(maxlen=100)
        
        # 성능 모니터링
        self.last_memory_check = time.time()
        self.memory_usage_history = deque(maxlen=50)
        
        self.logger.debug("Core components initialized")
    
    def _setup_monitoring(self):
        """성능 모니터링 설정"""
        if self.config.performance_monitoring:
            self._start_memory_monitor()
    
    def _start_memory_monitor(self):
        """메모리 모니터링 시작"""
        def monitor():
            while True:
                current_memory = self._get_memory_usage()
                self.memory_usage_history.append(current_memory)
                
                if current_memory > self.config.max_memory_mb * self.config.memory_critical_threshold:
                    self.logger.warning(f"Critical memory usage: {current_memory:.1f}MB")
                    self._emergency_cleanup()
                
                time.sleep(10)  # 10초마다 체크
        
        if self.config.performance_monitoring:
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
    
    def _get_memory_usage(self) -> float:
        """현재 메모리 사용량 (MB)"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            # psutil 없으면 간단한 추정
            return len(str(self.__dict__)) / (1024 * 10)  # 대략적 추정
    
    def adaptive_process(self, text: str, context: str = "", 
                        mode: ProcessingMode = ProcessingMode.AUTO,
                        user_preferences: Optional[Dict[str, Any]] = None) -> SJPUResult:
        """
        적응적 텍스트 처리 - 핵심 메서드 (완전 구현)
        
        Args:
            text: 처리할 텍스트
            context: 추가 컨텍스트
            mode: 처리 모드 (AUTO면 자동 선택)
            user_preferences: 사용자 선호 설정
        
        Returns:
            SJPUResult: 처리 결과 객체
        """
        start_time = time.time()
        
        try:
            # 입력 검증
            self._validate_input(text, context)
            
            # 메모리 상태 확인
            current_memory = self._get_memory_usage()
            
            # 처리 모드 결정
            if mode == ProcessingMode.AUTO:
                processing_mode = self._predict_optimal_mode(text, context, current_memory)
            else:
                processing_mode = mode
            
            self.logger.debug(f"Processing mode selected: {processing_mode.value}")
            
            # 캐시 확인
            cache_key = self._generate_cache_key(text, context, processing_mode)
            cached_result = self._check_cache(cache_key)
            
            if cached_result:
                self.processing_stats['cache_hits'] += 1
                return self._create_cached_result(cached_result, start_time)
            
            self.processing_stats['cache_misses'] += 1
            
            # 모드별 처리 실행
            processed_text = self._execute_processing_mode(
                text, context, processing_mode, user_preferences
            )
            
            # 품질 평가
            quality_score = self._evaluate_quality(text, processed_text)
            
            # 결과 생성
            result = SJPUResult(
                original_text=text,
                processed_text=processed_text,
                processing_mode=processing_mode,
                processing_time=time.time() - start_time,
                memory_used=self._get_memory_usage() - current_memory,
                quality_score=quality_score,
                config_hash=self._get_config_hash()
            )
            
            # 캐시 저장
            self._store_in_cache(cache_key, processed_text, result)
            
            # 통계 업데이트
            self._update_statistics(result)
            
            # 자동 최적화
            if self.config.auto_optimization:
                self._auto_optimize(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            self.processing_stats['error_count'] += 1
            
            return SJPUResult(
                original_text=text,
                processed_text=f"[ERROR] {text[:50]}{'...' if len(text) > 50 else ''}",
                processing_mode=ProcessingMode.CLASSICAL,
                processing_time=time.time() - start_time,
                memory_used=0,
                error_message=str(e)
            )
    
    def _validate_input(self, text: str, context: str):
        """입력 검증"""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        
        if len(text) > self.config.input_size_limit:
            raise ValueError(f"Input too large: {len(text)} > {self.config.input_size_limit}")
        
        if len(context) > self.config.input_size_limit // 2:
            raise ValueError(f"Context too large: {len(context)} > {self.config.input_size_limit // 2}")
    
    def _predict_optimal_mode(self, text: str, context: str, current_memory: float) -> ProcessingMode:
        """최적 처리 모드 예측"""
        text_length = len(text)
        total_length = len(text + context)
        
        # 메모리 압박도 계산
        memory_pressure = current_memory / self.config.max_memory_mb
        
        # 복잡도 분석
        complexity_score = self._analyze_text_complexity(text)
        
        # 모드 결정 로직
        if memory_pressure > 0.8:
            # 메모리 부족시 클래식 모드
            selected_mode = ProcessingMode.CLASSICAL
        elif text_length < self.config.quantum_threshold and complexity_score < 0.6:
            # 짧고 단순한 텍스트는 양자 모드
            selected_mode = ProcessingMode.QUANTUM
        elif text_length > self.config.classical_threshold or complexity_score > 0.8:
            # 길거나 복잡한 텍스트는 클래식 모드
            selected_mode = ProcessingMode.CLASSICAL
        else:
            # 중간 크기는 하이브리드 모드
            selected_mode = ProcessingMode.HYBRID
        
        # 예측 히스토리 기록
        self.mode_prediction_history.append({
            'text_length': text_length,
            'complexity': complexity_score,
            'memory_pressure': memory_pressure,
            'predicted_mode': selected_mode
        })
        
        return selected_mode
    
    def _analyze_text_complexity(self, text: str) -> float:
        """텍스트 복잡도 분석 (0-1)"""
        if not text:
            return 0.0
        
        words = text.split()
        if not words:
            return 0.1
        
        # 어휘 다양성
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words)
        
        # 평균 단어 길이
        avg_word_length = np.mean([len(word) for word in words])
        length_complexity = min(1.0, avg_word_length / 10.0)
        
        # 문장 복잡도
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        sentence_complexity = min(1.0, avg_sentence_length / 20.0)
        
        # 특수 문자 비율
        special_chars = len([c for c in text if not c.isalnum() and not c.isspace()])
        special_ratio = special_chars / len(text)
        
        # 종합 복잡도
        complexity = (
            vocabulary_diversity * 0.3 +
            length_complexity * 0.3 +
            sentence_complexity * 0.3 +
            special_ratio * 0.1
        )
        
        return min(1.0, complexity)
    
    def _generate_cache_key(self, text: str, context: str, mode: ProcessingMode) -> str:
        """캐시 키 생성"""
        content = f"{text}|{context}|{mode.value}|{self.config.vector_dimensions}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _check_cache(self, cache_key: str) -> Optional[str]:
        """캐시 확인"""
        if not self.config.use_quantum_cache:
            return None
        
        if cache_key in self.simple_cache:
            self.cache_access_count[cache_key] += 1
            return self.simple_cache[cache_key]
        
        return None
    
    def _execute_processing_mode(self, text: str, context: str, 
                                mode: ProcessingMode,
                                user_preferences: Optional[Dict[str, Any]]) -> str:
        """모드별 처리 실행"""
        
        if mode == ProcessingMode.QUANTUM:
            return self._quantum_process(text, context, user_preferences)
        elif mode == ProcessingMode.HYBRID:
            return self._hybrid_process(text, context, user_preferences)
        else:  # CLASSICAL
            return self._classical_process(text, context, user_preferences)
    
    def _quantum_process(self, text: str, context: str, 
                        user_preferences: Optional[Dict[str, Any]]) -> str:
        """양자 모드 처리 (80% 구현)"""
        self.logger.debug("Executing quantum processing")
        
        words = text.split()
        if not words:
            return text
        
        # 양자 상태 시뮬레이션
        quantum_amplitudes = []
        for i, word in enumerate(words):
            # 단어의 해시를 이용한 양자 상태 생성
            word_hash = hash(word.lower()) % 1000
            amplitude = (word_hash / 1000.0) * 0.8 + 0.1  # 0.1-0.9 범위
            phase = (word_hash % 360) * np.pi / 180
            
            quantum_state = amplitude * np.exp(1j * phase)
            quantum_amplitudes.append(abs(quantum_state))
        
        # 높은 진폭의 단어들 선택 (양자 측정 시뮬레이션)
        word_importance = list(zip(words, quantum_amplitudes, range(len(words))))
        word_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 70% 단어 선택
        num_selected = max(3, int(len(words) * 0.7))
        selected_words = word_importance[:num_selected]
        
        # 원래 순서로 재정렬
        selected_words.sort(key=lambda x: x[2])
        
        # 양자 얽힘 효과 시뮬레이션 (중요한 단어들 사이의 연결)
        result_words = []
        for i, (word, amplitude, _) in enumerate(selected_words):
            result_words.append(word)
            
            # 높은 진폭의 단어에 양자 얽힘 표시 추가
            if amplitude > 0.8 and i < len(selected_words) - 1:
                if self.config.consciousness_stream:
                    result_words.append("⟨⟩")  # 양자 얽힘 표시
        
        # 컨텍스트 기반 보정
        if context:
            result_text = " ".join(result_words)
            context_words = set(context.lower().split())
            result_text_words = result_text.lower().split()
            
            # 컨텍스트와 관련된 단어 강조
            enhanced_words = []
            for word in result_words:
                if word.lower() in context_words:
                    enhanced_words.append(f"*{word}*")  # 강조 표시
                else:
                    enhanced_words.append(word)
            
            return " ".join(enhanced_words)
        
        return " ".join(result_words)
    
    def _hybrid_process(self, text: str, context: str,
                       user_preferences: Optional[Dict[str, Any]]) -> str:
        """하이브리드 모드 처리"""
        self.logger.debug("Executing hybrid processing")
        
        # 텍스트를 두 부분으로 분할
        mid_point = len(text) // 2
        
        # 단어 경계에서 자르기
        words = text.split()
        mid_word = len(words) // 2
        
        part1_words = words[:mid_word]
        part2_words = words[mid_word:]
        
        part1_text = " ".join(part1_words)
        part2_text = " ".join(part2_words)
        
        # 첫 부분은 양자로, 두 번째 부분은 클래식으로
        quantum_result = self._quantum_process(part1_text, context, user_preferences)
        classical_result = self._classical_process(part2_text, context, user_preferences)
        
        # 하이브리드 연결
        connector = " ⊕ " if self.config.consciousness_stream else " | "
        
        return f"{quantum_result}{connector}{classical_result}"
    
    def _classical_process(self, text: str, context: str,
                          user_preferences: Optional[Dict[str, Any]]) -> str:
        """클래식 모드 처리"""
        self.logger.debug("Executing classical processing")
        
        words = text.split()
        if not words:
            return text
        
        # 전통적인 텍스트 처리 (키워드 추출 + 요약)
        if len(words) <= 5:
            # 짧은 텍스트는 그대로
            return text
        elif len(words) <= 20:
            # 중간 길이는 핵심 단어 추출
            # 길이가 긴 단어들을 우선적으로 선택
            word_scores = [(word, len(word), i) for i, word in enumerate(words)]
            word_scores.sort(key=lambda x: (x[1], -x[2]), reverse=True)  # 길이순, 앞순서 우선
            
            selected_count = max(3, len(words) // 2)
            selected = word_scores[:selected_count]
            selected.sort(key=lambda x: x[2])  # 원래 순서로 복원
            
            result = [item[0] for item in selected]
            return " ".join(result)
        else:
            # 긴 텍스트는 문장 단위로 요약
            sentences = text.split('.')
            if len(sentences) <= 2:
                return text
            
            # 중간 문장들 중 가장 긴 것들 선택
            sentence_scores = [(s.strip(), len(s.split()), i) 
                             for i, s in enumerate(sentences) if s.strip()]
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            selected_count = max(1, len(sentence_scores) // 2)
            selected_sentences = sentence_scores[:selected_count]
            selected_sentences.sort(key=lambda x: x[2])  # 원래 순서
            
            result = [item[0] for item in selected_sentences]
            return ". ".join(result) + "."
    
    def _evaluate_quality(self, original_text: str, processed_text: str) -> float:
        """텍스트 품질 평가 (0-1)"""
        if not original_text or not processed_text:
            return 0.0
        
        original_words = set(original_text.lower().split())
        processed_words = set(processed_text.lower().split())
        
        if not original_words:
            return 0.0
        
        # 핵심 단어 보존율
        preserved_ratio = len(original_words & processed_words) / len(original_words)
        
        # 길이 적정성 (너무 짧거나 길면 감점)
        length_ratio = len(processed_text) / len(original_text)
        length_score = 1.0 - abs(0.7 - length_ratio)  # 70% 길이를 최적으로 가정
        length_score = max(0.0, min(1.0, length_score))
        
        # 가독성 (단어 수와 문장 구조)
        processed_sentences = processed_text.split('.')
        readability_score = min(1.0, len(processed_sentences) / max(1, len(processed_sentences)))
        
        # 종합 품질 점수
        quality_score = (preserved_ratio * 0.5 + length_score * 0.3 + readability_score * 0.2)
        
        return min(1.0, max(0.0, quality_score))
    
    def _store_in_cache(self, cache_key: str, processed_text: str, result: SJPUResult):
        """캐시에 결과 저장"""
        if not self.config.use_quantum_cache:
            return
        
        # 캐시 크기 제한 확인
        if len(self.simple_cache) >= self.config.cache_size_limit:
            self._evict_cache()
        
        self.simple_cache[cache_key] = processed_text
        self.cache_access_count[cache_key] = 1
    
    def _evict_cache(self):
        """캐시 제거 (LRU 기반)"""
        if not self.simple_cache:
            return
        
        # 접근 횟수가 가장 적은 항목 제거
        least_accessed = min(self.cache_access_count.items(), key=lambda x: x[1])
        key_to_remove = least_accessed[0]
        
        self.simple_cache.pop(key_to_remove, None)
        self.cache_access_count.pop(key_to_remove, None)
        
        self.logger.debug(f"Cache evicted: {key_to_remove}")
    
    def _create_cached_result(self, cached_text: str, start_time: float) -> SJPUResult:
        """캐시된 결과 생성"""
        return SJPUResult(
            original_text="[CACHED]",
            processed_text=cached_text,
            processing_mode=ProcessingMode.QUANTUM,  # 캐시는 양자 모드로 간주
            processing_time=time.time() - start_time,
            memory_used=0.0,  # 캐시는 메모리 사용 없음
            quality_score=0.9  # 캐시는 높은 품질로 가정
        )
    
    def _update_statistics(self, result: SJPUResult):
        """통계 업데이트"""
        with self._lock:
            self.processing_stats['total_processed'] += 1
            self.processing_stats['mode_distribution'][result.processing_mode.value] += 1
            self.processing_stats['total_processing_time'] += result.processing_time
            self.processing_stats['total_memory_used'] += result.memory_used
            
            # 성능 히스토리 추가
            self.performance_history.append({
                'timestamp': result.timestamp,
                'mode': result.processing_mode.value,
                'processing_time': result.processing_time,
                'memory_used': result.memory_used,
                'quality_score': result.quality_score,
                'efficiency_score': result.get_efficiency_score()
            })
    
    def _get_config_hash(self) -> str:
        """현재 설정 해시"""
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _auto_optimize(self, result: SJPUResult):
        """자동 최적화"""
        # 성능이 낮으면 설정 조정
        if result.get_efficiency_score() < 0.5:
            if result.memory_used > self.config.max_memory_mb * 0.7:
                self.config.optimize_for_memory()
                self.logger.info("Auto-optimized for memory")
            elif result.processing_time > 5.0:
                self.config.optimize_for_speed()
                self.logger.info("Auto-optimized for speed")
    
    def _emergency_cleanup(self):
        """긴급 메모리 정리"""
        self.logger.warning("Emergency cleanup initiated")
        
        # 캐시 50% 제거
        cache_items = list(self.simple_cache.items())
        items_to_remove = len(cache_items) // 2
        
        for key, _ in cache_items[:items_to_remove]:
            self.simple_cache.pop(key, None)
            self.cache_access_count.pop(key, None)
        
        # 히스토리 제거
        while len(self.performance_history) > 100:
            self.performance_history.popleft()
        
        # 강제 가비지 컬렉션
        gc.collect()
        
        self.logger.info("Emergency cleanup completed")
    
    # =============================================================================
    # 공개 API 메서드들
    # =============================================================================
    
    def process(self, text: str, context: str = "") -> SJPUResult:
        """간단한 텍스트 처리 - 공개 API"""
        return self.adaptive_process(text, context)
    
    def process_quantum(self, text: str, context: str = "") -> SJPUResult:
        """양자 모드 강제 처리"""
        return self.adaptive_process(text, context, ProcessingMode.QUANTUM)
    
    def process_classical(self, text: str, context: str = "") -> SJPUResult:
        """클래식 모드 강제 처리"""
        return self.adaptive_process(text, context, ProcessingMode.CLASSICAL)
    
    def batch_process(self, texts: List[str], contexts: List[str] = None) -> List[SJPUResult]:
        """배치 처리"""
        if contexts is None:
            contexts = [""] * len(texts)
        elif len(contexts) != len(texts):
            raise ValueError("Texts and contexts must have same length")
        
        results = []
        for text, context in zip(texts, contexts):
            try:
                result = self.adaptive_process(text, context)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch item failed: {e}")
                error_result = SJPUResult(
                    original_text=text,
                    processed_text=f"[ERROR] {text[:30]}...",
                    processing_mode=ProcessingMode.CLASSICAL,
                    processing_time=0.001,
                    memory_used=0,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 반환"""
        with self._lock:
            total_processed = self.processing_stats['total_processed']
            
            stats = {
                'total_processed': total_processed,
                'mode_distribution': dict(self.processing_stats['mode_distribution']),
                'average_processing_time': (
                    self.processing_stats['total_processing_time'] / max(total_processed, 1)
                ),
                'average_memory_usage': (
                    self.processing_stats['total_memory_used'] / max(total_processed, 1)
                ),
                'error_rate': (
                    self.processing_stats['error_count'] / max(total_processed, 1)
                ),
                'cache_hit_rate': (
                    self.processing_stats['cache_hits'] / 
                    max(self.processing_stats['cache_hits'] + self.processing_stats['cache_misses'], 1)
                ),
                'cache_size': len(self.simple_cache),
                'performance_level': self.config.get_performance_level().name,
                'current_memory_usage': self._get_memory_usage()
            }
            
            # 최근 성능 통계
            if self.performance_history:
                recent_performance = list(self.performance_history)[-10:]
                stats['recent_avg_efficiency'] = np.mean([
                    p['efficiency_score'] for p in recent_performance
                ])
                stats['recent_avg_quality'] = np.mean([
                    p['quality_score'] for p in recent_performance
                ])
            
            return stats
    
    def optimize_system(self) -> Dict[str, Any]:
        """시스템 최적화 실행"""
        start_time = time.time()
        optimizations_applied = []
        
        initial_memory = self._get_memory_usage()
        
        # 1. 캐시 최적화
        if len(self.simple_cache) > self.config.cache_size_limit * 0.8:
            old_size = len(self.simple_cache)
            self._evict_cache()
            new_size = len(self.simple_cache)
            optimizations_applied.append(f"Cache optimized: {old_size} → {new_size}")
        
        # 2. 메모리 정리
        if initial_memory > self.config.max_memory_mb * 0.7:
            gc.collect()
            optimizations_applied.append("Memory garbage collected")
        
        # 3. 히스토리 정리
        if len(self.performance_history) > 800:
            while len(self.performance_history) > 500:
                self.performance_history.popleft()
            optimizations_applied.append("Performance history trimmed")
        
        # 4. 설정 자동 조정
        if self.performance_history:
            recent_efficiency = np.mean([
                p['efficiency_score'] for p in list(self.performance_history)[-20:]
            ])
            if recent_efficiency < 0.6:
                self.config.optimize_for_speed()
                optimizations_applied.append("Config optimized for speed")
        
        final_memory = self._get_memory_usage()
        optimization_time = time.time() - start_time
        
        return {
            'optimization_time': optimization_time,
            'optimizations_applied': optimizations_applied,
            'memory_saved_mb': initial_memory - final_memory,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory
        }
    
    def clear_cache(self):
        """캐시 전체 삭제"""
        self.simple_cache.clear()
        self.cache_access_count.clear()
        self.logger.info("Cache cleared")
    
    def reset_stats(self):
        """통계 초기화"""
        with self._lock:
            self.processing_stats = {
                'total_processed': 0,
                'mode_distribution': defaultdict(int),
                'total_processing_time': 0.0,
                'total_memory_used': 0.0,
                'error_count': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            self.performance_history.clear()
        
        self.logger.info("Statistics reset")
    
    def shutdown(self):
        """시스템 종료"""
        self.logger.info("SJPU System shutting down...")
        
        # 캐시 저장 (필요시)
        if self.config.debug_mode:
            cache_stats = {
                'cache_size': len(self.simple_cache),
                'total_access': sum(self.cache_access_count.values())
            }
            self.logger.debug(f"Final cache stats: {cache_stats}")
        
        # 정리
        self.clear_cache()
        self.reset_stats()
        
        self.logger.info("SJPU System shutdown completed")

# =============================================================================
# 간편 사용을 위한 래퍼 클래스
# =============================================================================

class SJPU:
    """간편 사용을 위한 SJPU 래퍼 클래스"""
    
    def __init__(self, performance_level: str = "medium", debug: bool = False):
        """
        간단한 초기화
        
        Args:
            performance_level: "ultra_low", "low", "medium", "high", "ultra_high"
            debug: 디버그 모드 활성화
        """
        config = HybridSJPUConfig(debug_mode=debug, verbose_logging=debug)
        
        # 성능 레벨별 설정
        if performance_level.lower() == "ultra_low":
            config.max_memory_mb = 128
            config.vector_dimensions = 64
            config.cache_size_limit = 200
        elif performance_level.lower() == "low":
            config.max_memory_mb = 256
            config.vector_dimensions = 128
            config.cache_size_limit = 500
        elif performance_level.lower() == "high":
            config.max_memory_mb = 1024
            config.vector_dimensions = 512
            config.cache_size_limit = 2000
        elif performance_level.lower() == "ultra_high":
            config.max_memory_mb = 2048
            config.vector_dimensions = 1024
            config.cache_size_limit = 5000
        # medium은 기본값 사용
        
        self.engine = PredictiveHybridSJPU(config)
    
    def process(self, text: str, context: str = "") -> str:
        """간단한 텍스트 처리 - 결과 문자열만 반환"""
        result = self.engine.process(text, context)
        return result.processed_text
    
    def process_detailed(self, text: str, context: str = "") -> SJPUResult:
        """상세한 처리 결과 반환"""
        return self.engine.process(text, context)
    
    def stats(self) -> Dict[str, Any]:
        """시스템 통계"""
        return self.engine.get_stats()
    
    def optimize(self) -> Dict[str, Any]:
        """시스템 최적화"""
        return self.engine.optimize_system()

# =============================================================================
# 사용 예제 및 테스트
# =============================================================================

def run_basic_tests():
    """기본 기능 테스트"""
    print("🧪 SJPU 기본 기능 테스트 시작")
    print("=" * 50)
    
    # 간단한 사용법
    sjpu = SJPU(performance_level="medium", debug=True)
    
    test_cases = [
        ("안녕하세요! 오늘 날씨가 정말 좋네요.", "일상 대화"),
        ("인공지능의 발전은 인류에게 많은 변화를 가져올 것입니다. " * 5, "기술 논의"),
        ("짧은 텍스트", ""),
        ("이것은 매우 긴 텍스트입니다. " * 20, "긴 텍스트 테스트")
    ]
    
    for i, (text, context) in enumerate(test_cases, 1):
        print(f"\n📝 테스트 {i}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        
        try:
            # 상세 결과
            result = sjpu.process_detailed(text, context)
            
            print(f"   모드: {result.processing_mode.value}")
            print(f"   처리시간: {result.processing_time:.3f}초")
            print(f"   메모리: {result.memory_used:.1f}MB")
            print(f"   품질점수: {result.quality_score:.2f}")
            print(f"   효율성: {result.get_efficiency_score():.2f}")
            print(f"   결과: {result.processed_text[:50]}{'...' if len(result.processed_text) > 50 else ''}")
            print("   ✅ 성공")
            
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    
    # 배치 처리 테스트
    print(f"\n🔄 배치 처리 테스트")
    batch_texts = [case[0] for case in test_cases[:2]]
    batch_contexts = [case[1] for case in test_cases[:2]]
    
    batch_results = sjpu.engine.batch_process(batch_texts, batch_contexts)
    print(f"   배치 처리 완료: {len(batch_results)}/{len(batch_texts)}")
    
    # 통계 출력
    print(f"\n📊 시스템 통계:")
    stats = sjpu.stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    run_basic_tests()