"""모델링용 사용자 정의 모델."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC


class TanimotoSVC(ClassifierMixin, BaseEstimator):
    """Tanimoto(Jaccard) 커널 SVM. 분자 핑거프린트(0/1)의 유사도를 직접 커널로 쓴다.

    입력은 0보다 크면 1로 이진화한다. 트리 모델과 다른 방식으로 구조 유사성을 쓰므로 앙상블 다양성에 유리하다.
    """

    def __init__(self, C=1.0):
        self.C = C

    @staticmethod
    def _kernel(A, B, sa, sb):
        inter = A @ B.T
        return inter / (sa[:, None] + sb[None, :] - inter + 1e-9)

    def fit(self, X, y):
        A = (np.asarray(X) > 0).astype(np.float32)
        self.A_, self.sa_ = A, A.sum(1)
        K = self._kernel(A, A, self.sa_, self.sa_)
        self.svc_ = SVC(C=self.C, kernel="precomputed", probability=True, random_state=0).fit(K, y)
        self.classes_ = self.svc_.classes_
        return self

    def predict_proba(self, X):
        B = (np.asarray(X) > 0).astype(np.float32)
        return self.svc_.predict_proba(self._kernel(B, self.A_, B.sum(1), self.sa_))

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
