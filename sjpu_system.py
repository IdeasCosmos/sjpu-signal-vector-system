import numpy as np
from scipy.stats import entropy
from scipy.signal import convolve
from scipy.special import kl_div
from control import TransferFunction, forced_response
import mpmath
import sympy as sp
import warnings
from sklearn.cluster import SpectralClustering
import time

try:
    import faiss
except ImportError:
    faiss = None

class SJPUConfig:
    MAX_LAYERS = 20
    DEFAULT_SAMPLES = 1000  # 테스트 및 실전 속도 개선을 위해 100000에서 줄임
    BANDWIDTH = 0.05
    DAMPING = 0.1
    MAX_DB_SIZE = 10000
    EPSILON = 1e-10
    # FAISS 최적화 파라미터
    N_CLUSTERS = 32  # IVFFlat 클러스터 수 (검색 속도 최적화)
    N_PROBE = 8  # 검색 시 탐색 클러스터 수 (정확도 vs 속도 균형)
    HNSW_M = 32  # HNSW 그래프 연결 수 (메모리 vs 속도)
    HNSW_EF_SEARCH = 64  # 검색 시 탐색 깊이

class SJPUVectorSystem:
    def __init__(self, dim=100, config=None):
        self.dim = dim
        self.config = config or SJPUConfig()
        self._initialize_database()
        self.benchmark_results = {'add_times': [], 'query_times': []}  # 벤치마크 결과 저장

    def _initialize_database(self):
        """내부 데이터베이스 초기화. faiss 없으면 numpy 폴백."""
        self.metadata = []
        self.vectors = []
        self.use_faiss = False  # 기본값 False로 설정 (faiss 없어도 동작 보장)

        if faiss is not None:
            try:
                # FAISS 성능 최적화: IndexIVFFlat 사용 (클러스터링으로 검색 속도 향상)
                quantizer = faiss.IndexFlatL2(self.dim)
                self.knowledge_db = faiss.IndexIVFFlat(quantizer, self.dim, self.config.N_CLUSTERS)
                self.knowledge_db.nprobe = self.config.N_PROBE  # 검색 최적화 파라미터
                self.use_faiss = True
            except Exception as e:
                warnings.warn(f"FAISS 초기화 오류: {e}. numpy 폴백 모드로 전환합니다.")
                self.knowledge_db = np.empty((0, self.dim))
        else:
            warnings.warn(
                "FAISS를 사용할 수 없습니다. numpy 폴백 모드로 전환합니다.\n"
                "벡터 검색 성능이 떨어질 수 있습니다. 'faiss-cpu'를 requirements.txt에 추가 후 설치하세요."
            )
            self.knowledge_db = np.empty((0, self.dim))

        # faiss 사용 여부와 관계없이 knowledge_db shape 보장
        if not self.use_faiss and self.knowledge_db.size == 0:
            self.knowledge_db = np.empty((0, self.dim))

        # FAISS 학습 (초기화 시 클러스터 학습 필요 시)
        if self.use_faiss and faiss is not None:
            # 초기 학습 데이터 생성 (임시 벡터로 학습)
            train_data = np.random.random((self.config.N_CLUSTERS * 10, self.dim)).astype('float32')
            self.knowledge_db.train(train_data)

    def validate_vector(self, vec):
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        if vec.shape[0] != self.dim:
            old_dim = vec.shape[0]
            if old_dim > self.dim:
                vec = vec[:self.dim]
            else:
                vec = np.pad(vec, (0, self.dim - old_dim), 'constant')
            warnings.warn(f"벡터 크기 {old_dim}에서 {self.dim}으로 조정됨")
        if not np.isfinite(vec).all():
            vec = np.nan_to_num(vec)
            warnings.warn("NaN/Inf 대체")
        return vec

    def generate_vector(self, vec_type='gaussian'):
        if vec_type == 'uniform':
            vec = np.ones(self.dim) / np.sqrt(self.dim)
        elif vec_type == 'gaussian':
            vec = np.exp(-np.linspace(-3, 3, self.dim)**2 / 2)
            vec /= np.linalg.norm(vec)
        elif vec_type == 'sparse':
            probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
            vec = np.zeros(self.dim)
            vec[:5] = np.sqrt(probs)
        elif vec_type == 'impulse':
            vec = np.zeros(self.dim)
            vec[0] = 1.0
        else:
            vec = np.random.normal(0, 1, self.dim)
            vec /= np.linalg.norm(vec)  # 명시적 정규화 추가
        return self.validate_vector(vec)

    def quantum_collapse_metrics(self, vec):
        vec = self.validate_vector(vec)
        probs = np.abs(vec)**2
        probs /= np.sum(probs) + self.config.EPSILON
        outcomes = np.random.choice(self.dim, size=self.config.DEFAULT_SAMPLES, p=probs)
        hist, _ = np.histogram(outcomes, bins=self.dim, density=True)
        ent = entropy(probs, base=np.e)
        kl = np.sum(kl_div(hist + self.config.EPSILON, probs + self.config.EPSILON))
        unique = np.sum(hist > 1e-6)
        corr = np.corrcoef(probs, hist)[0, 1] if np.std(hist) > 0 else np.nan
        return {'entropy': ent, 'kl': kl, 'unique': unique, 'corr': corr}

    def riemann_zeta_transform(self, vec, s_real=0.5, s_imag=0.0):
        vec = self.validate_vector(vec)
        k = np.arange(1, self.dim + 1)
        powers = 1 / np.power(k, s_real)  # NumPy approximation for speed
        transformed = vec * powers
        amp = np.mean(np.abs(transformed)) / (np.mean(np.abs(vec)) + self.config.EPSILON)
        energy = np.linalg.norm(transformed)**2 / (np.linalg.norm(vec)**2 + self.config.EPSILON)
        return transformed, amp, energy

    def bell_transform(self, vec, depth=0.5):
        vec = self.validate_vector(vec)
        order = min(20, max(1, int(depth * 10)))
        bell_coeffs = np.array([float(sp.bell(k)) for k in range(order + 1)])
        bell_coeffs /= bell_coeffs.sum() + self.config.EPSILON
        smoothed = convolve(vec, bell_coeffs, mode='same')
        vec_var = np.var(np.diff(vec)) + self.config.EPSILON
        smooth_var = np.var(np.diff(smoothed)) + self.config.EPSILON
        improve = vec_var / smooth_var
        noise_red = min(100, (np.var(vec - smoothed) / (np.var(vec) + self.config.EPSILON)) * 100)  # 100 초과 방지
        return smoothed, improve, noise_red

    def critical_line_modulation(self, vec, max_layers=None):
        vec = self.validate_vector(vec)
        max_layers = max_layers or self.config.MAX_LAYERS
        ent = entropy(np.abs(vec)**2 + self.config.EPSILON)
        layers = min(max_layers, max(1, int(ent * 2)))
        modulated = vec.copy()
        max_coh = 0
        best_layer = 0
        for layer in range(1, layers + 1):
            s_imag = layer * 1.0
            modulated, _, _ = self.riemann_zeta_transform(modulated, s_imag=s_imag)
            phase = np.angle(modulated + self.config.EPSILON * 1j)
            coh = np.abs(np.mean(np.exp(1j * phase)))
            if coh > max_coh:
                max_coh = coh
                best_layer = layer
        stability = np.std(modulated) / (np.std(vec) + self.config.EPSILON)
        return modulated, best_layer, stability

    def resonance_pattern(self, vec, bandwidth=None, damping=None):
        vec = self.validate_vector(vec)
        bandwidth = bandwidth or self.config.BANDWIDTH
        damping = damping or self.config.DAMPING
        try:
            q = 1 / bandwidth
            sys = TransferFunction([q], [1, damping, q**2])
            t = np.arange(self.dim)
            t_out, filtered, _ = forced_response(sys, T=t, U=vec)
            if len(filtered) != self.dim:
                filtered = np.interp(np.arange(self.dim), np.linspace(0, self.dim - 1, len(filtered)), filtered)
            filtered = np.nan_to_num(filtered)
            eff = np.linalg.norm(filtered)**2 / (np.linalg.norm(vec)**2 + self.config.EPSILON)
            return filtered, q, eff
        except Exception as e:
            warnings.warn(f"Resonance error: {e}")
            return vec, 1.0, 1.0

    def adaptive_process_pipeline(self, vec_type='sparse', adaptive=True):
        vec = self.generate_vector(vec_type)
        if adaptive:
            similar, dists = self.query_db(vec, k=5)
            if len(similar) > 0:
                vectors = [m['vector'] for m in similar if 'vector' in m]
                vectors.append(vec)
                affinity = np.corrcoef(vectors)
                sc = SpectralClustering(n_clusters=2, affinity='precomputed')
                labels = sc.fit_predict(affinity)
                assoc_score = np.mean(np.abs(labels[:-1]))
                self.config.BANDWIDTH = 0.05 / (1 + assoc_score)
                self.config.DAMPING = 0.1 * (1 + assoc_score)
        qc_metrics = self.quantum_collapse_metrics(vec)
        zeta_vec, amp, energy = self.riemann_zeta_transform(vec)
        bell_vec, improve, noise_red = self.bell_transform(zeta_vec)
        mod_vec, coh_layer, stab = self.critical_line_modulation(bell_vec)
        res_vec, q, eff = self.resonance_pattern(mod_vec)
        meta = {
            'type': vec_type, 'qc_metrics': qc_metrics, 'amp': amp, 'energy': energy,
            'improve': improve, 'noise_red': noise_red, 'coh_layer': coh_layer,
            'stab': stab, 'q': q, 'eff': eff, 'vector': vec
        }
        self.add_to_db(res_vec, meta)
        results = {'amp': amp, 'energy': energy, 'improve': improve, 'noise_red': noise_red,
                   'coh_layer': coh_layer, 'q': q, 'eff': eff, 'stab': stab}
        return res_vec, results

    def add_to_db(self, vec, meta):
        start_time = time.time()  # 벤치마크 시작
        vec = self.validate_vector(vec)
        vec_f32 = vec.astype(np.float32)
        if self.use_faiss:
            if self.knowledge_db.ntotal >= self.config.MAX_DB_SIZE:
                self.metadata.pop(0)
                self.vectors.pop(0)
                self.knowledge_db = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dim), self.dim, self.config.N_CLUSTERS)
                self.knowledge_db.nprobe = self.config.N_PROBE
                if self.vectors:
                    self.knowledge_db.add(np.array(self.vectors).astype(np.float32))
            self.knowledge_db.add(vec_f32.reshape(1, -1))
            self.vectors.append(vec_f32)
            self.metadata.append(meta)
        else:
            if len(self.knowledge_db) >= self.config.MAX_DB_SIZE:
                self.knowledge_db = self.knowledge_db[1:]
                self.metadata.pop(0)
            self.knowledge_db = np.vstack((self.knowledge_db, vec_f32.reshape(1, -1))) if self.knowledge_db.size else vec_f32.reshape(1, -1)
            self.metadata.append(meta)
        end_time = time.time()
        self.benchmark_results['add_times'].append(end_time - start_time)  # 추가 시간 기록

    def query_db(self, query_vec, k=3):
        start_time = time.time()  # 벤치마크 시작
        query_vec = self.validate_vector(query_vec)
        query_f32 = query_vec.astype(np.float32).reshape(1, -1)
        if self.use_faiss:
            if self.knowledge_db.ntotal == 0:
                return [], []
            k = min(k, self.knowledge_db.ntotal)
            D, I = self.knowledge_db.search(query_f32, k)
            return [self.metadata[i] for i in I[0]], D[0]
        else:
            if len(self.knowledge_db) == 0:
                return [], []
            dists = np.linalg.norm(self.knowledge_db - query_f32, axis=1)
            idx = np.argsort(dists)[:min(k, len(dists))]
            return [self.metadata[i] for i in idx], dists[idx]
        end_time = time.time()
        self.benchmark_results['query_times'].append(end_time - start_time)  # 검색 시간 기록

    def get_system_stats(self):
        db_size = self.knowledge_db.ntotal if self.use_faiss else len(self.knowledge_db)
        return {'dim': self.dim, 'db_size': db_size, 'max_db_size': self.config.MAX_DB_SIZE, 'using_faiss': self.use_faiss, 'metadata_count': len(self.metadata)}

    def benchmark_db(self, num_operations=10):
        """벡터 DB 간략 벤치마크: 추가 및 검색 시간 평균 계산."""
        print("벤치마크 시작: 추가/검색 시간 측정 ({}회 반복)".format(num_operations))
        for _ in range(num_operations):
            vec = self.generate_vector('gaussian')
            meta = {'test': 'benchmark'}
            self.add_to_db(vec, meta)  # 추가 벤치마크
            self.query_db(vec, k=3)  # 검색 벤치마크
        add_avg = np.mean(self.benchmark_results['add_times']) if self.benchmark_results['add_times'] else 0
        query_avg = np.mean(self.benchmark_results['query_times']) if self.benchmark_results['query_times'] else 0
        print("평균 추가 시간: {:.6f} 초".format(add_avg))
        print("평균 검색 시간: {:.6f} 초".format(query_avg))
        print("벤치마크 완료. using_faiss: {}".format(self.use_faiss))
        # 결과 초기화 (재사용 시)
        self.benchmark_results = {'add_times': [], 'query_times': []}

if __name__ == "__main__":
    system = SJPUVectorSystem(dim=100)
    # 벤치마크 예시 실행
    system.benchmark_db(num_operations=10)