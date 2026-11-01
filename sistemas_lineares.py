import numpy as np


def _preparar_sistema(A, b, x0, tol, max_iter):
    matriz = np.asarray(A, dtype=float)
    vetor = np.asarray(b, dtype=float)

    if matriz.ndim != 2 or matriz.shape[0] == 0 or matriz.shape[0] != matriz.shape[1]:
        raise ValueError("A matriz A deve ser quadrada e nao vazia.")
    if vetor.ndim != 1 or vetor.shape[0] != matriz.shape[0]:
        raise ValueError("O vetor b deve ter o mesmo tamanho da matriz A.")
    if not np.all(np.isfinite(matriz)) or not np.all(np.isfinite(vetor)):
        raise ValueError("Os valores do sistema devem ser finitos.")
    if np.any(np.diag(matriz) == 0):
        raise ValueError("A diagonal da matriz A nao pode conter zeros.")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("A tolerancia deve ser um numero positivo e finito.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("O numero maximo de iteracoes deve ser um inteiro positivo.")

    if x0 is None:
        aproximacao = np.zeros(vetor.shape[0])
    else:
        aproximacao = np.asarray(x0, dtype=float)
        if aproximacao.shape != vetor.shape:
            raise ValueError("A aproximacao inicial deve ter o mesmo tamanho de b.")
        if not np.all(np.isfinite(aproximacao)):
            raise ValueError("Os valores da aproximacao inicial devem ser finitos.")

    return matriz, vetor, aproximacao


def jacobi(A, b, x0=None, tol=1e-5, max_iter=100):
    A, b, x = _preparar_sistema(A, b, x0, tol, max_iter)
    n = len(b)
    for k in range(max_iter):
        x_novo = np.zeros(n)
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x_novo[i] = (b[i] - soma) / A[i][i]
        if np.linalg.norm(x_novo - x) < tol:
            return x_novo, k + 1
        x = x_novo
    return x, max_iter


def gauss_seidel(A, b, x0=None, tol=1e-5, max_iter=100):
    A, b, x = _preparar_sistema(A, b, x0, tol, max_iter)
    n = len(b)
    for k in range(max_iter):
        x_antigo = x.copy()
        for i in range(n):
            soma = sum(A[i][j] * x[j] for j in range(n) if i != j)
            x[i] = (b[i] - soma) / A[i][i]
        if np.linalg.norm(x - x_antigo) < tol:
            return x, k + 1
    return x, max_iter