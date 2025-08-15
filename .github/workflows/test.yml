import numpy as np
import pytest
from sjpu_system import SJPUVectorSystem, SJPUConfig

# 필요한 패키지 경고 무시
import warnings
warnings.filterwarnings("ignore")

# 테스트용 설정
@pytest.fixture
def system():
    config = SJPUConfig()
    config.MAX_DB_SIZE = 5  # 작은 값으로 테스트 속도 개선
    return SJPUVectorSystem(dim=5, config=config)

# 1. 초기화 테스트
def test_init(system):
    assert system.dim == 5
    assert isinstance(system.config, SJPUConfig)
    assert (system.knowledge_db.ntotal == 0 if system.use_faiss else len(system.knowledge_db) == 0)
    assert len(system.metadata) == 0
    assert len(system.vectors) == 0

# 2. 벡터 생성 테스트
@pytest.mark.parametrize("vec_type", ["uniform", "gaussian", "sparse", "impulse", "random"])
def test_generate_vector(system, vec_type):
    vec = system.generate_vector(vec_type)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == system.dim
    assert np.all(np.isfinite(vec))
    assert np.abs(np.linalg.norm(vec) - 1.0) < 1e-10  # 정규화 확인

# 3. 벡터 검증 테스트
def test_validate_vector(system):
    vec = np.random.normal(0, 1, 3)  # dim(5)보다 작은 벡터
    validated = system.validate_vector(vec)
    assert validated.shape[0] == system.dim
    assert np.all(np.isfinite(validated))

    # NaN 포함 테스트
    vec_with_nan = np.array([1.0, np.nan, 2.0])
    validated_nan = system.validate_vector(vec_with_nan)
    assert np.all(np.isfinite(validated_nan))

# 4. 양자 붕괴 메트릭 테스트
def test_quantum_collapse_metrics(system):
    vec = system.generate_vector("gaussian")
    metrics = system.quantum_collapse_metrics(vec)
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["entropy", "kl", "unique", "corr"])
    assert metrics["entropy"] >= 0
    assert not np.isnan(metrics["corr"]) or np.isfinite(metrics["corr"])

# 5. 리만 제타 변환 테스트
def test_riemann_zeta_transform(system):
    vec = system.generate_vector("uniform")
    transformed, amp, energy = system.riemann_zeta_transform(vec)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == vec.shape
    assert amp > 0
    assert energy > 0

# 6. 벨 변환 테스트
def test_bell_transform(system):
    vec = system.generate_vector("gaussian")
    smoothed, improve, noise_red = system.bell_transform(vec)
    assert isinstance(smoothed, np.ndarray)
    assert smoothed.shape == vec.shape
    assert improve >= 1.0  # 평활화로 분산 감소
    assert 0 <= noise_red <= 100

# 7. 임계선 변조 테스트
def test_critical_line_modulation(system):
    vec = system.generate_vector("sparse")
    modulated, best_layer, stability = system.critical_line_modulation(vec)
    assert isinstance(modulated, np.ndarray)
    assert modulated.shape == vec.shape
    assert 0 <= best_layer <= system.config.MAX_LAYERS
    assert stability > 0

# 8. 공진 패턴 테스트
def test_resonance_pattern(system):
    vec = system.generate_vector("impulse")
    filtered, q, eff = system.resonance_pattern(vec)
    assert isinstance(filtered, np.ndarray)
    assert filtered.shape == vec.shape
    assert q > 0
    assert 0 <= eff <= 1.0

# 9. 적응형 파이프라인 테스트
def test_adaptive_process_pipeline(system):
    vec_type = "uniform"
    processed, metrics = system.adaptive_process_pipeline(vec_type, adaptive=True)
    assert isinstance(processed, np.ndarray)
    assert processed.shape == (system.dim,)
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ["amp", "energy", "improve", "noise_red", "coh_layer", "q", "eff", "stab"])
    assert metrics["amp"] > 0
    assert metrics["energy"] > 0

# 10. 데이터베이스 추가 테스트
def test_add_to_db(system):
    vec = system.generate_vector("gaussian")
    meta = {"test": "meta"}
    system.add_to_db(vec, meta)
    assert system.knowledge_db.ntotal == 1 if system.use_faiss else len(system.knowledge_db) == 1
    assert len(system.metadata) == 1
    assert system.metadata[0] == meta

# 11. 데이터베이스 쿼리 테스트
def test_query_db(system):
    vec1 = system.generate_vector("gaussian")
    meta1 = {"test": "meta1"}
    system.add_to_db(vec1, meta1)
    vec2 = system.generate_vector("uniform")
    results, dists = system.query_db(vec2, k=1)
    assert len(results) == 1
    assert len(dists) == 1
    assert dists[0] >= 0

# 12. 시스템 통계 테스트
def test_get_system_stats(system):
    stats = system.get_system_stats()
    assert isinstance(stats, dict)
    assert all(key in stats for key in ["dim", "db_size", "max_db_size", "using_faiss", "metadata_count"])
    assert stats["dim"] == system.dim
    assert stats["db_size"] == 0
    assert stats["max_db_size"] == system.config.MAX_DB_SIZE

if __name__ == "__main__":
    pytest.main([__file__])