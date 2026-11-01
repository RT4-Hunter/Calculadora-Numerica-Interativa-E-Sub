import re

from flask import Flask, render_template, request

from interpolacao.interpolacao import interpolacao_lagrange
from interpolacao.sistemas_lineares import gauss_seidel, jacobi


app = Flask(__name__)


def _ler_numeros(texto: str, descricao: str) -> list[float]:
    try:
        numeros = [float(item) for item in re.split(r"[\s,;]+", texto.strip()) if item]
    except ValueError as erro:
        raise ValueError(f"Ha um valor invalido em {descricao}.") from erro
    if not numeros:
        raise ValueError(f"Informe os valores de {descricao}.")
    return numeros


def _ler_matriz(texto: str) -> list[list[float]]:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    if not linhas:
        raise ValueError("Informe as linhas da matriz A.")
    return [_ler_numeros(linha, "uma linha da matriz A") for linha in linhas]


def _calcular_sistema(metodo: str) -> tuple[str, str]:
    matriz = _ler_matriz(request.form.get("matriz", ""))
    vetor = _ler_numeros(request.form.get("vetor", ""), "o vetor b")
    aproximacao_texto = request.form.get("aproximacao", "").strip()
    aproximacao = _ler_numeros(aproximacao_texto, "a aproximacao inicial") if aproximacao_texto else None

    try:
        tolerancia = float(request.form.get("tolerancia", "1e-5"))
        max_iteracoes = int(request.form.get("max_iteracoes", "100"))
    except ValueError as erro:
        raise ValueError("Tolerancia e iteracoes devem ser valores numericos validos.") from erro

    funcao = jacobi if metodo == "jacobi" else gauss_seidel
    solucao, iteracoes = funcao(
        matriz,
        vetor,
        x0=aproximacao,
        tol=tolerancia,
        max_iter=max_iteracoes,
    )
    valores = ", ".join(f"x{indice + 1} = {valor:.8g}" for indice, valor in enumerate(solucao))
    return f"Solucoes: {valores}", f"Iteracoes realizadas: {iteracoes}"


def _calcular_lagrange() -> tuple[str, str]:
    linhas = [linha.strip() for linha in request.form.get("pontos", "").splitlines() if linha.strip()]
    if not linhas:
        raise ValueError("Informe pelo menos um ponto (x, y).")

    pontos: list[tuple[float, float]] = []
    for indice, linha in enumerate(linhas, start=1):
        coordenadas = _ler_numeros(linha, f"o ponto {indice}")
        if len(coordenadas) != 2:
            raise ValueError(f"O ponto {indice} deve conter exatamente x e y.")
        pontos.append((coordenadas[0], coordenadas[1]))

    try:
        x = float(request.form.get("x_interpolacao", ""))
    except ValueError as erro:
        raise ValueError("Informe um valor numerico para x.") from erro

    resultado = interpolacao_lagrange(pontos, x)
    return f"Resultado: P({x:g}) = {resultado:.8g}", ""


@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    detalhe = None
    erro = None
    metodo = request.form.get("metodo", "jacobi")

    if request.method == "POST":
        try:
            if metodo in {"jacobi", "gauss_seidel"}:
                resultado, detalhe = _calcular_sistema(metodo)
            elif metodo == "lagrange":
                resultado, detalhe = _calcular_lagrange()
            else:
                raise ValueError("Selecione um metodo valido.")
        except (ValueError, OverflowError, TypeError) as excecao:
            erro = str(excecao)

    return render_template(
        "index.html",
        metodo=metodo,
        resultado=resultado,
        detalhe=detalhe,
        erro=erro,
        dados=request.form if request.method == "POST" else {},
    )


if __name__ == "__main__":
    app.run(debug=False)
