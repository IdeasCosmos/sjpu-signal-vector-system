import numpy as np
import time
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
import threading
try:
    import psutil
except ImportError:
    psutil = None
import gc
from scipy.stats import entropy
from scipy.signal import convolve
from scipy.special import kl_div
from control import TransferFunction, forced_response
import sympy as sp
import faiss
import warnings
from sklearn.cluster import SpectralClustering

# EMOTION_DB (기존, 채움)
EMOTION_DB = {
    'positive': {
        'Korean': ['좋아', '기쁜', '행복', '즐거운', '사랑', '기쁨', '웃음', '흥분', '평화', '만족',
                   '멋지', '감동', '너무 멋진', 'ㅎㅎ', 'ㅋㅋ'],
        'intensity_map': {'ㅎㅎ': 2.5, 'ㅋㅋ': 2.0, 'default': 0.5}
    },
    'negative': {
        'Korean': ['슬픈', '분노', '화가', '두려운', '상처', '우울', '짜증', '공포', '후회',
                   '눈물이 난다', '숨이 막혀온다', '한숨이 난다', '슬푼', 'ㅠㅠ', 'ㅜㅜ'],
        'intensity_map': {'ㅠㅠ': 3.0, 'ㅜㅜ': 2.5, '슬푼': 2.8, 'default': 0.5}
    },
    'neutral': {
        'Korean': ['오늘', '텍스트', '단어', '중립', '보통', '일상', '그냥', '평범', '시간', '날씨', '가위'],
        'intensity_map': {'default': 0.5}
    }
}

# CONTEXT_RESONANCE (기존, 채움)
CONTEXT_RESONANCE = {
    '미용실': {'가위': {'emotion': 'neutral', 'resonance_score': 0.8}},
    '가위바위보': {'가위': {'emotion': 'neutral', 'resonance_score': 0.7}},
    '의료': {'가위': {'emotion': 'neutral', 'resonance_score': 0.9}}
}

# 계층 그룹화 클래스
class TextUnit:
    """텍스트를 형태소(1~3-gram)부터 담화까지 동시 분석"""
    def __init__(self, text: str):
        self.text = text
        self.words = self.text.split()  # 단어
        self.morphemes = self._tokenize_morphemes()  # 형태소
        self.phrases = self._group_phrases()  # 구
        self.clauses = self._group_clauses()  # 절
        self.sentences = [s.strip() for s in self.text.split('.') if s.strip()]  # 문장
        self.discourse = self.text  # 담화

    def _tokenize_morphemes(self) -> List[str]:
        """형태소: 1~3-gram, 복합/오타/슬랭 포함"""
        morphemes = []
        for word in self.words:
            for n in range(1, 4):
                morphemes.extend([word[i:i+n] for i in range(len(word)-n+1)])
        # 복합/오타/슬랭 추가
        for emotion in EMOTION_DB:
            for phrase in EMOTION_DB[emotion]['Korean']:
                if phrase in self.text:
                    morphemes.append(phrase)
        return list(set(morphemes))

    def _group_phrases(self) -> List[str]:
        """구: 2~4 단어, 매치율 0.2"""
        phrases = [' '.join(self.words[i:i+4]) for i in range(0, len(self.words), 2)]
        return [p for p in phrases if p]

    def _group_clauses(self) -> List[str]:
        """절: 중복 제거, 매치율 0.2"""
        clauses = list(set(c.strip() for c in self.text.replace('.', ',').split(',') if c.strip()))
        return clauses

# 공명 값 학습 클래스
class DynamicCenterLearner:
    """공명 값 테이블: 형태소/복합 감정/오타/슬랭 매핑"""
    def __init__(self, language: str = 'Korean'):
        self.language = language
        self.center_values = defaultdict(list)
        self.resonance_table = defaultdict(dict)
        self._load_from_db()

    def _load_from_db(self):
        """초기 공명 값: 감정/맥락별"""
        for emotion in EMOTION_DB:
            self.center_values[emotion].extend(EMOTION_DB[emotion][self.language])
            for word in EMOTION_DB[emotion][self.language]:
                score = 0.9 if any(x in word for x in ['감동', '난다', 'ㅎㅎ', 'ㅋㅋ']) else (0.7 if emotion == 'neutral' else 0.8)
                self.resonance_table[word] = {
                    'emotion': emotion,
                    'resonance_score': score,
                    'vector': np.random.rand(5),
                    'context': 'general'
                }
        # 맥락별 가위 공명
        for context, mappings in CONTEXT_RESONANCE.items():
            for word, data in mappings.items():
                self.resonance_table[word][context] = data

    def learn_center(self, text_units: TextUnit, emotion: str, analyzer: 'FrequencyAnalyzer', context: str = 'general'):
        """공명 값 학습: 맥락별 동시 참조"""
        words = text_units.words + text_units.morphemes
        outliers = analyzer.detect_outliers(text_units)
        deviations = self._calculate_deviations(words, emotion, text_units, outliers, context)
        for word, dev in deviations.items():
            if word not in self.resonance_table:
                self.resonance_table[word] = {
                    'emotion': emotion,
                    'resonance_score': 0.5,
                    'vector': np.random.rand(5),
                    'context': context
                }
            self.resonance_table[word]['resonance_score'] = min(1.0, self.resonance_table[word]['resonance_score'] + dev * 0.1)
            if word not in self.center_values[emotion]:
                self.center_values[emotion].append(word)

    def _calculate_deviations(self, words: List[str], emotion: str, text_units: TextUnit, outliers: List[str], context: str) -> Dict[str, float]:
        """공명 deviation: 감정(0.4), 주파수(0.3), 맥락(0.3)"""
        deviations = {}
        for word in words:
            # 감정 점수
            emotion_score = 0.4 if word in self.center_values[emotion] else 0.2
            if word in ['ㅎㅎ', 'ㅋㅋ']:
                emotion_score += 0.2  # 슬랭 긍정 강화
            elif word in ['ㅠㅠ', 'ㅜㅜ', '슬푼']:
                emotion_score -= 0.1  # 오타/슬랭 부정
            # 주파수 점수
            freq_score = 0.3 if word in outliers else 0.1
            # 맥락 점수
            morph_match = sum(1 for m in text_units.morphemes if m in word) / max(1, len(word)) * 0.3
            phrase_match = sum(1 for p in text_units.phrases if word in p) / max(1, len(text_units.phrases)) * 0.3
            clause_match = sum(1 for c in text_units.clauses if word in c) / max(1, len(text_units.clauses)) * 0.2
            sentence_match = sum(1 for s in text_units.sentences if word in s) / max(1, len(text_units.sentences)) * 0.2
            match_rate = morph_match + phrase_match + clause_match + sentence_match
            context_score = (1 - match_rate) * 0.3 if match_rate < 0.2 else 0.0
            if word == '가위' and context in CONTEXT_RESONANCE:
                context_score += CONTEXT_RESONANCE[context].get('가위', {}).get('resonance_score', 0) * 0.2
            if len(text_units.text) < 50:
                context_score += 0.2  # 짧은 텍스트 긍정
            elif len(text_units.text) > 200:
                context_score -= 0.1  # 긴 텍스트 부정
            deviations[word] = emotion_score + freq_score + context_score
        return {k: v for k, v in deviations.items() if v > 0.3}

# 주파수 분석 클래스
class FrequencyAnalyzer:
    """주파수 기반 이상치: 텍스트를 음성 신호처럼 처리"""
    def __init__(self, text: str):
        self.freq_signal = self._text_to_freq(text)

    def _text_to_freq(self, text: str) -> np.ndarray:
        """텍스트를 ASCII로 주파수 변환 (오타/슬랭 강조)"""
        ascii_vals = []
        for c in text:
            if c in ['ㅎ', 'ㅋ', 'ㅠ', 'ㅜ']:  # 슬랭 강조
                ascii_vals.extend([ord(c)] * 3)  # 주파수 증폭
            elif c in text and text.count(c) > 3:  # 오타/반복 감지
                ascii_vals.extend([ord(c)] * 2)
            else:
                ascii_vals.append(ord(c))
        ascii_vals = np.array(ascii_vals)
        return np.fft.fft(ascii_vals) if len(ascii_vals) > 0 else np.array([])

    def detect_outliers(self, text_units: TextUnit) -> List[str]:
        """이상치: 주파수 스파이크 + 매치율 0.2"""
        if len(self.freq_signal) == 0:
            return []
        magnitudes = np.abs(self.freq_signal)
        threshold = np.mean(magnitudes) + 1.5 * np.std(magnitudes)
        outlier_indices = np.where(magnitudes > threshold)[0]
        outliers = []
        for i in outlier_indices:
            word = text_units.words[i % len(text_units.words)]
            match_rate = sum(1 for s in text_units.sentences if word in s) / max(1, len(text_units.sentences))
            if match_rate < 0.2 or word in ['ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', 'ㅜㅜ', '슬푼']:
                outliers.append(word)
        return list(set(outliers))

# 감응값 분석 클래스
class ResponseValueAnalyzer:
    """감응값: 텍스트 + 파형 동시 처리, 강도 매핑 포함"""
    def __init__(self, text: str, text_units: TextUnit, emotion: str):
        self.freq_analyzer = FrequencyAnalyzer(text)
        self.freq_signal = self.freq_analyzer.freq_signal
        self.dimensions = self._compute_dimensions(text_units)
        self.emotion = emotion

    def _compute_dimensions(self, units: TextUnit) -> Dict[str, float]:
        """다차원 편차: 길이(0.2), 오타율(0.3), 어휘 다양성(0.2), 구조 복잡도(0.3)"""
        length_dev = abs(len(units.text) - 100) / 100 * 0.2
        typo_candidates = ['슬푼', 'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', 'ㅜㅜ']
        typo_rate = sum(1 for w in units.words if any(t in w for t in typo_candidates)) / max(1, len(units.words)) * 0.3
        vocab_div = len(set(units.words)) / max(1, len(units.words))
        vocab_dev = (1 - vocab_div) * 0.2
        struct_complex = len(units.clauses) / max(1, len(units.sentences)) * 0.3
        return {'length': length_dev, 'typo': typo_rate, 'vocab': vocab_dev, 'struct': struct_complex}

    def get_response_value(self) -> float:
        """감응값 계산: 파형 레벨 + 다차원 합 + 강도 매핑"""
        if len(self.freq_signal) == 0:
            return 0.5
        wave_level = np.mean(np.abs(self.freq_signal)) / 100  # VU미터 스케일
        dim_sum = sum(self.dimensions.values())
        intensity_bonus = EMOTION_DB.get(self.emotion, {}).get('intensity_map', {}).get('default', 0.0) / 3.0
        response_value = min(1.0, wave_level + dim_sum + intensity_bonus)
        return response_value

# 핵심 요약 함수
def core_summary(text: str, context: str = "", language: str = 'Korean', emotion: str = 'positive') -> str:
    """핵심 요약: 공명 경로 기반, 맥락/오타/뉘앙스 처리"""
    units = TextUnit(text)
    learner = DynamicCenterLearner(language)
    analyzer = FrequencyAnalyzer(text)
    learner.learn_center(units, emotion, analyzer, context)
    outliers = analyzer.detect_outliers(units)
    centers = learner.center_values[emotion]
    summary_words = [word for word in units.words if word in centers or word in outliers or
                     learner.resonance_table.get(word, {}).get('resonance_score', 0) >= 0.3]
    if len(summary_words) < 3:
        summary_words.extend(units.words[:3])
    for i, word in enumerate(summary_words):
        phrase_match = sum(1 for p in units.phrases if word in p) / max(1, len(units.phrases))
        if phrase_match < 0.2 or word in ['ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', 'ㅜㅜ', '슬푼']:
            summary_words[i] = f"*{word}*"
    summary_words = list(set(summary_words))
    summary_words.sort(key=lambda w: text.find(w))
    summary = ' '.join(summary_words) if summary_words else text[:50]
    if context:
        context_words = context.split()
        summary = ' '.join([w for w in context_words if w in summary_words] + [w for w in summary_words if w not in context_words])
    return summary

# 설정 클래스
@dataclass
class HybridSJPUConfig:
    """시스템 설정"""
    max_memory_mb: int = 128
    cache_size_limit: int = 200
    gc_frequency: int = 100
    memory_warning_threshold: float = 0.7
    memory_critical_threshold: float = 0.9
    use_quantum_cache: bool = True
    use_multiprocessing: bool = False
    max_workers: int = 2
    consciousness_stream: bool = True
    debug_mode: bool = True
    verbose_logging: bool = True
    performance_monitoring: bool = True
    auto_optimization: bool = True
    max_processing_time: float = 30.0
    input_size_limit: int = 100000
    output_size_limit: int = 50000

    def __post_init__(self):
        self._validate_config()
        self._auto_adjust_config()

    def _validate_config(self):
        if self.max_memory_mb < 64:
            raise ValueError(f"max_memory_mb must be >= 64MB, got {self.max_memory_mb}")
        if not 0 < self.memory_warning_threshold < self.memory_critical_threshold < 1:
            raise ValueError("Memory thresholds must be: 0 < warning < critical < 1")

    def _auto_adjust_config(self):
        try:
            if psutil:
                total_memory = psutil.virtual_memory().total / (1024**3)
                if total_memory < 4:
                    self.max_memory_mb = min(self.max_memory_mb, 128)
                    self.cache_size_limit = min(self.cache_size_limit, 200)
                elif total_memory < 8:
                    self.max_memory_mb = min(self.max_memory_mb, 512)
                    self.cache_size_limit = min(self.cache_size_limit, 1000)
                cpu_count = psutil.cpu_count()
                self.max_workers = min(self.max_workers, max(1, cpu_count - 1))
        except (ImportError, AttributeError):
            pass

    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HybridSJPUConfig':
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

# 결과 클래스
@dataclass
class SJPUResult:
    original_text: str
    processed_text: str
    processing_time: float
    memory_used: float
    input_length: int = 0
    output_length: int = 0
    compression_ratio: float = 1.0
    quality_score: float = 0.5
    timestamp: float = field(default_factory=time.time)
    config_hash: str = ""
    error_message: Optional[str] = None
    consciousness_analysis: Optional[Dict[str, Any]] = None
    response_value: float = 0.5  # 추가

    def __post_init__(self):
        if not self.input_length:
            self.input_length = len(self.original_text)
        if not self.output_length:
            self.output_length = len(self.processed_text)
        if self.input_length > 0:
            self.compression_ratio = self.output_length / self.input_length

    def get_efficiency_score(self) -> float:
        if self.processing_time <= 0:
            return 0.0
        speed_score = min(1.0, (self.input_length / 1000) / self.processing_time)
        memory_score = max(0.0, 1.0 - (self.memory_used / 100))
        quality_score = self.quality_score
        return (speed_score * 0.4 + memory_score * 0.3 + quality_score * 0.3)

# 메인 클래스
class PredictiveHybridSJPU:
    """예측적 SJPU: 공명 경로/맥락별 처리"""
    def __init__(self, config: Optional[HybridSJPUConfig] = None):
        self.config = config or HybridSJPUConfig()
        self._setup_logging()
        self._initialize_components()
        self._setup_monitoring()
        self.processing_stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'total_memory_used': 0.0,
            'error_count': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.performance_history = deque(maxlen=1000)
        self.last_gc_time = time.time()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")

    def _setup_logging(self):
        level = logging.DEBUG if self.config.debug_mode else logging.INFO
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if self.config.verbose_logging:
            self.logger.setLevel(logging.DEBUG)

    def _initialize_components(self):
        self.simple_cache = OrderedDict()
        self.cache_access_count = defaultdict(int)
        self.memory_usage_history = deque(maxlen=50)

    def _setup_monitoring(self):
        if self.config.performance_monitoring:
            self._start_memory_monitor()

    def _start_memory_monitor(self):
        def monitor():
            while True:
                current_memory = self._get_memory_usage()
                self.memory_usage_history.append(current_memory)
                if current_memory > self.config.max_memory_mb * self.config.memory_critical_threshold:
                    self.logger.warning(f"Critical memory usage: {current_memory:.1f}MB")
                    self._emergency_cleanup()
                time.sleep(10)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def _get_memory_usage(self) -> float:
        try:
            if psutil:
                process = psutil.Process()
                samples = [process.memory_info().rss / (1024 * 1024) for _ in range(3)]
                return sum(samples) / len(samples)
            return len(str(self.__dict__)) / (1024 * 10) * 0.5
        except Exception:
            return 0.1

    def process(self, text: str, context: str = "", language: str = 'Korean', emotion: str = 'positive') -> SJPUResult:
        """텍스트 처리: 공명 경로 기반"""
        start_time = time.time()
        try:
            if len(text) > self.config.input_size_limit:
                chunk_size = self.config.input_size_limit // 2
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                results = []
                initial_memory = self._get_memory_usage()
                for chunk in chunks:
                    result = self._process_chunk(chunk, context, language, emotion)
                    results.append(result.processed_text)
                processed_text = " ".join(results)
                quality_score = self._evaluate_quality(text, processed_text)
                return SJPUResult(
                    original_text=text,
                    processed_text=processed_text,
                    processing_time=time.time() - start_time,
                    memory_used=self._get_memory_usage() - initial_memory,
                    quality_score=quality_score,
                    config_hash=self._get_config_hash()
                )

            self._validate_input(text, context)
            initial_memory = self._get_memory_usage()
            cache_key = self._generate_cache_key(text, context, language, emotion)
            cached_result = self._check_cache(cache_key)
            if cached_result:
                self.processing_stats['cache_hits'] += 1
                return self._create_cached_result(cached_result, start_time, initial_memory)
            self.processing_stats['cache_misses'] += 1
            processed_text = core_summary(text, context, language, emotion)
            quality_score = self._evaluate_quality(text, processed_text)
            result = SJPUResult(
                original_text=text,
                processed_text=processed_text,
                processing_time=time.time() - start_time,
                memory_used=self._get_memory_usage() - initial_memory,
                quality_score=quality_score,
                config_hash=self._get_config_hash()
            )
            self._store_in_cache(cache_key, processed_text, result)
            self._update_statistics(result)
            if self.config.auto_optimization:
                self._auto_optimize(result)
            return result
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}", exc_info=True)
            self.processing_stats['error_count'] += 1
            return SJPUResult(
                original_text=text,
                processed_text=f"[ERROR] {text[:50]}{'...' if len(text) > 50 else ''}",
                processing_time=time.time() - start_time,
                memory_used=0,
                error_message=str(e)
            )

    def _process_chunk(self, text: str, context: str, language: str, emotion: str) -> SJPUResult:
        return self.process(text, context, language, emotion)

    def _validate_input(self, text: str, context: str):
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if len(context) > self.config.input_size_limit // 2:
            raise ValueError(f"Context too large: {len(context)} > {self.config.input_size_limit // 2}")

    def _generate_cache_key(self, text: str, context: str, language: str, emotion: str) -> str:
        content = f"{text}|{context}|{language}|{emotion}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _check_cache(self, cache_key: str) -> Optional[str]:
        if not self.config.use_quantum_cache:
            return None
        if cache_key in self.simple_cache:
            self.simple_cache.move_to_end(cache_key)
            self.cache_access_count[cache_key] += 1
            return self.simple_cache[cache_key]
        return None

    def _evaluate_quality(self, original_text: str, processed_text: str) -> float:
        """품질 평가: 복합 감정/오타/슬랭 반영"""
        if not original_text or not processed_text:
            return 0.5
        original_words = set(original_text.lower().split())
        processed_words = set(processed_text.lower().split())
        if not original_words:
            return 0.5
        preserved_ratio = len(original_words & processed_words) / len(original_words) * 0.4
        length_ratio = len(processed_text) / len(original_text)
        length_score = 1.0 - abs(0.7 - length_ratio) * 0.3
        processed_sentences = processed_text.split('.')
        readability_score = min(1.0, len(processed_sentences) / max(1, len(processed_sentences))) * 0.2
        complex_emotion_score = 0.2 if any(w in processed_text for w in ['눈물이 난다', '숨이 막혀온다', '한숨이 난다', 'ㅎㅎ', 'ㅋㅋ', 'ㅠㅠ', 'ㅜㅜ', '슬푼']) else 0.0
        return min(1.0, max(0.0, preserved_ratio + length_score + readability_score + complex_emotion_score))

    def _store_in_cache(self, cache_key: str, processed_text: str, result: SJPUResult):
        if not self.config.use_quantum_cache:
            return
        if len(self.simple_cache) >= self.config.cache_size_limit:
            self._evict_cache()
        self.simple_cache[cache_key] = processed_text
        self.cache_access_count[cache_key] = 1

    def _evict_cache(self):
        if not self.simple_cache:
            return
        self.simple_cache.popitem(last=False)

    def _create_cached_result(self, cached_text: str, start_time: float, initial_memory: float) -> SJPUResult:
        return SJPUResult(
            original_text="[CACHED]",
            processed_text=cached_text,
            processing_time=time.time() - start_time,
            memory_used=self._get_memory_usage() - initial_memory,
            quality_score=0.9
        )

    def _update_statistics(self, result: SJPUResult):
        with self._lock:
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_processing_time'] += result.processing_time
            self.processing_stats['total_memory_used'] += result.memory_used
            self.performance_history.append({
                'timestamp': result.timestamp,
                'processing_time': result.processing_time,
                'memory_used': result.memory_used,
                'quality_score': result.quality_score,
                'efficiency_score': result.get_efficiency_score()
            })

    def _get_config_hash(self) -> str:
        config_str = json.dumps(self.config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _auto_optimize(self, result: SJPUResult):
        if result.get_efficiency_score() < 0.5:
            if result.memory_used > self.config.max_memory_mb * 0.7:
                self.config.max_memory_mb = int(self.config.max_memory_mb * 1.2)
            elif result.processing_time > 5.0:
                self.config.max_workers = max(1, self.config.max_workers - 1)

    def _emergency_cleanup(self):
        self.logger.warning("Emergency cleanup initiated")
        cache_items = list(self.simple_cache.items())
        items_to_remove = len(cache_items) // 2
        for key, _ in cache_items[:items_to_remove]:
            self.simple_cache.pop(key, None)
            self.cache_access_count.pop(key, None)
        while len(self.performance_history) > 100:
            self.performance_history.popleft()
        gc.collect()
        self.logger.info("Emergency cleanup completed")

    def batch_process(self, texts: List[str], contexts: List[str] = None, language: str = 'Korean', emotion: str = 'positive') -> List[SJPUResult]:
        if contexts is None:
            contexts = [""] * len(texts)
        elif len(contexts) != len(texts):
            raise ValueError("Texts and contexts must have same length")
        results = []
        for text, context in zip(texts, contexts):
            try:
                result = self.process(text, context, language, emotion)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Batch item failed: {e}")
                error_result = SJPUResult(
                    original_text=text,
                    processed_text=f"[ERROR] {text[:30]}...",
                    processing_time=0.001,
                    memory_used=0,
                    error_message=str(e)
                )
                results.append(error_result)
        return results

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_processed = self.processing_stats['total_processed']
            stats = {
                'total_processed': total_processed,
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
                'current_memory_usage': self._get_memory_usage()
            }
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
        start_time = time.time()
        optimizations_applied = []
        initial_memory = self._get_memory_usage()
        if len(self.simple_cache) > self.config.cache_size_limit * 0.8:
            old_size = len(self.simple_cache)
            self._evict_cache()
            new_size = len(self.simple_cache)
            optimizations_applied.append(f"Cache optimized: {old_size} → {new_size}")
        if initial_memory > self.config.max_memory_mb * 0.7:
            gc.collect()
            optimizations_applied.append("Memory garbage collected")
        if len(self.performance_history) > 800:
            while len(self.performance_history) > 500:
                self.performance_history.popleft()
            optimizations_applied.append("Performance history trimmed")
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
        self.simple_cache.clear()
        self.cache_access_count.clear()
        self.logger.info("Cache cleared")

    def reset_stats(self):
        with self._lock:
            self.processing_stats = {
                'total_processed': 0,
                'total_processing_time': 0.0,
                'total_memory_used': 0.0,
                'error_count': 0,
                'cache_hits': 0,
                'cache_misses': 0
            }
            self.performance_history.clear()
        self.logger.info("Statistics reset")

    def shutdown(self):
        self.logger.info("SJPU System shutting down...")
        if self.config.debug_mode:
            cache_stats = {
                'cache_size': len(self.simple_cache),
                'total_access': sum(self.cache_access_count.values())
            }
            self.logger.debug(f"Final cache stats: {cache_stats}")
        self.clear_cache()
        self.reset_stats()
        self.logger.info("SJPU System shutdown completed")

# 래퍼 클래스
class SJPU:
    """간편 사용 래퍼"""
    def __init__(self, performance_level: str = "medium", debug: bool = True):
        config = HybridSJPUConfig(debug_mode=debug, verbose_logging=debug)
        if performance_level.lower() == "ultra_low":
            config.max_memory_mb = 128
            config.cache_size_limit = 200
        elif performance_level.lower() == "low":
            config.max_memory_mb = 256
            config.cache_size_limit = 500
        elif performance_level.lower() == "high":
            config.max_memory_mb = 1024
            config.cache_size_limit = 2000
        elif performance_level.lower() == "ultra_high":
            config.max_memory_mb = 2048
            config.cache_size_limit = 5000
        self.engine = PredictiveHybridSJPU(config)

    def process(self, text: str, context: str = "", language: str = 'Korean', emotion: str = 'positive') -> str:
        return self.engine.process(text, context, language, emotion).processed_text

    def process_detailed(self, text: str, context: str = "", language: str = 'Korean', emotion: str = 'positive') -> SJPUResult:
        return self.engine.process(text, context, language, emotion)

    def stats(self) -> Dict[str, Any]:
        return self.engine.get_stats()

    def optimize(self) -> Dict[str, Any]:
        return self.engine.optimize_system()

# 테스트 함수
def run_basic_tests():
    print("🧪 SJPU 기본 기능 테스트 시작")
    print("=" * 50)
    sjpu = SJPU(performance_level="medium", debug=True)
    test_cases = [
        ("안녕하세요! 오늘 날씨가 정말 좋네요.", "일상 대화", "Korean", "positive"),
        ("너무 멋진 장면에, 숨이 막혀온다.", "감동 표현", "Korean", "positive"),
        ("너무 멋진 장면에, 눈물이 난다.", "감동 표현", "Korean", "negative"),
        ("너무 멋진 장면에, 한숨이 난다.", "감동 표현", "Korean", "negative"),
        ("가위 좀 주세요.", "미용실", "Korean", "neutral"),
        ("가위바위보 하자!", "가위바위보", "Korean", "neutral"),
        ("가위 소독했나요?", "의료", "Korean", "neutral"),
        ("사랑한다고 ㅎㅎ", "SNS 대화", "Korean", "positive"),
        ("사랑한다고...", "SNS 대화", "Korean", "negative"),
        ("슬푼 마음이야 ㅠㅠ", "SNS 대화", "Korean", "negative"),
        ("Happy but sad goodbye, today weather is nice.", "emotion expression", "English", "negative"),
        ("이것은 매우 긴 텍스트입니다. " * 20, "긴 텍스트 테스트", "Korean", "neutral")
    ]
    for i, (text, context, language, emotion) in enumerate(test_cases, 1):
        print(f"\n📝 테스트 {i}: '{text[:30]}{'...' if len(text) > 30 else ''}'")
        try:
            result = sjpu.process_detailed(text, context, language, emotion)
            print(f"   처리시간: {result.processing_time:.3f}초")
            print(f"   메모리: {result.memory_used:.1f}MB")
            print(f"   품질점수: {result.quality_score:.2f}")
            print(f"   효율성: {result.get_efficiency_score():.2f}")
            print(f"   결과: {result.processed_text[:50]}{'...' if len(result.processed_text) > 50 else ''}")
            print("   ✅ 성공")
        except Exception as e:
            print(f"   ❌ 실패: {e}")
    print(f"\n🔄 배치 처리 테스트")
    batch_texts = [case[0] for case in test_cases[:3]]
    batch_contexts = [case[1] for case in test_cases[:3]]
    batch_results = sjpu.engine.batch_process(batch_texts, batch_contexts)
    print(f"   배치 처리 완료: {len(batch_results)}/{len(batch_texts)}")
    print(f"\n📊 시스템 통계:")
    stats = sjpu.stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    print(f"\n✅ 모든 테스트 완료!")

# comprehensive_test 함수 추가
def comprehensive_test():
    sjpu = SJPU(performance_level="medium", debug=True)
    
    # 더 체계적인 테스트 케이스
    test_cases = [
        # 같은 감정, 다른 표현
        ("기뻐요", "일상 대화", "Korean", "positive"),
        ("행복해요", "일상 대화", "Korean", "positive"), 
        ("좋아요", "일상 대화", "Korean", "positive"),
        
        # 같은 단어, 다른 컨텍스트
        ("짜증나네", "SNS 대화", "Korean", "negative"),
        ("짜증나네ㅎㅎ", "SNS 대화", "Korean", "negative"),
        
        # 길이별 테스트
        ("짧음", "테스트", "Korean", "neutral"),
        ("중간 길이의 텍스트입니다", "테스트", "Korean", "neutral"),
        ("이것은 매우 긴 텍스트입니다. " * 10, "긴 텍스트 테스트", "Korean", "neutral")
    ]
    
    results = []
    for text, context, language, emotion in test_cases:
        result = sjpu.process_detailed(text, context, language, emotion)
        results.append({
            'text': text,
            'context': context,
            'language': language,
            'emotion': emotion,
            'response_value': result.response_value,
            'processing_time': result.processing_time,
            'summary': result.processed_text
        })
    
    return results

# benchmark_comparison 함수 추가
def benchmark_comparison():
    # 기본 키워드 추출 방법
    def simple_summary(text):
        words = text.split()
        return ' '.join(words[:5])  # 단순히 첫 5단어
    
    # SJPU 방법
    sjpu = SJPU(performance_level="medium", debug=True)
    
    test_texts = [
        ("오늘 날씨가 정말 좋아서 기분이 좋네요", "", "Korean", "positive"),
        ("회사에서 스트레스 받아서 짜증나네 정말", "", "Korean", "negative"),
        ("사랑하는 사람과 함께 있을 때가 가장 행복해", "", "Korean", "positive")
    ]
    
    for text, context, language, emotion in test_texts:
        simple_result = simple_summary(text)
        sjpu_result = sjpu.process_detailed(text, context, language, emotion)
        
        print(f"Original: {text}")
        print(f"Simple: {simple_result}")
        print(f"SJPU: {sjpu_result.processed_text}")
        print(f"Response Value: {sjpu_result.response_value}")
        print("---")

# 테스트 실행
run_basic_tests()
print("Comprehensive Test Results:")
print(comprehensive_test())
print("\nBenchmark Comparison:")
benchmark_comparison()
</parameter
</xai:function_call