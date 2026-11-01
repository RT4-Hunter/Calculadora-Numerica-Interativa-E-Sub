from collections.abc import Sequence
from math import isfinite


def _validar_pontos(pontos: Sequence[tuple[float, float]]) -> None:
    if not pontos:
        raise ValueError("Informe pelo menos um ponto.")

    for ponto in pontos:
        if len(ponto) != 2:
            raise ValueError("Cada ponto deve conter um par (x, y).")
        if not isfinite(ponto[0]) or not isfinite(ponto[1]):
            raise ValueError("As coordenadas dos pontos devem ser finitas.")

    xs = [ponto[0] for ponto in pontos]
    if len(set(xs)) != len(xs):
        raise ValueError("Os valores de x dos pontos devem ser distintos.")


def interpolacao_lagrange(pontos: Sequence[tuple[float, float]], x: float) -> float:
    if not pontos:
        raise ValueError("Informe pelo menos um ponto.")
    if not isfinite(x):
        raise ValueError("x deve ser um numero finito.")

    _validar_pontos(pontos)

    resultado = 0.0
    for i, (xi, yi) in enumerate(pontos):
        base = 1.0
        for j, (xj, _) in enumerate(pontos):
            if i != j:
                base *= (x - xj) / (xi - xj)
        resultado += yi * base

    if not isfinite(resultado):
        raise ValueError("O resultado da interpolacao nao e finito.")
    return resultado


def _coeficientes_polinomio(
    pontos: Sequence[tuple[float, float]],
) -> list[float]:
    coeficientes = [0.0] * len(pontos)

    for i, (xi, yi) in enumerate(pontos):
        base = [1.0]
        denominador = 1.0
        for j, (xj, _) in enumerate(pontos):
            if i == j:
                continue

            nova_base = [0.0] * (len(base) + 1)
            for grau, coeficiente in enumerate(base):
                nova_base[grau] -= xj * coeficiente
                nova_base[grau + 1] += coeficiente
            base = nova_base
            denominador *= xi - xj
            if denominador == 0 or not isfinite(denominador):
                raise ValueError("Nao foi possivel representar os coeficientes do polinomio.")

        for grau, coeficiente in enumerate(base):
            coeficientes[grau] += yi * coeficiente / denominador

    if not all(isfinite(coeficiente) for coeficiente in coeficientes):
        raise ValueError("Nao foi possivel representar os coeficientes do polinomio.")
    return coeficientes


def _formatar_polinomio(coeficientes: Sequence[float]) -> str:
    tolerancia = max((abs(coeficiente) for coeficiente in coeficientes), default=0.0)
    tolerancia *= 1e-10
    termos: list[tuple[bool, str]] = []

    for grau, coeficiente in reversed(list(enumerate(coeficientes))):
        if abs(coeficiente) <= tolerancia:
            continue

        magnitude = format(abs(coeficiente), ".6g")
        if grau == 0:
            termo = magnitude
        else:
            variavel = "x" if grau == 1 else f"x^{grau}"
            termo = variavel if magnitude == "1" else f"{magnitude}{variavel}"
        termos.append((coeficiente < 0, termo))

    if not termos:
        return "P(x) = 0"

    expressao = ("-" if termos[0][0] else "") + termos[0][1]
    for negativo, termo in termos[1:]:
        expressao += (" - " if negativo else " + ") + termo
    return f"P(x) = {expressao}"


def exibir_grafico_lagrange(pontos: Sequence[tuple[float, float]]) -> None:
    _validar_pontos(pontos)

    import matplotlib.pyplot as plt

    coeficientes = _coeficientes_polinomio(pontos)
    menor_x = min(x for x, _ in pontos)
    maior_x = max(x for x, _ in pontos)
    intervalo = maior_x - menor_x
    if intervalo == 0:
        intervalo = 2.0
    margem = intervalo * 0.1

    inicio = menor_x - margem
    fim = maior_x + margem
    quantidade_amostras = 400
    valores_x = [
        inicio + (fim - inicio) * indice / (quantidade_amostras - 1)
        for indice in range(quantidade_amostras)
    ]
    valores_y = [interpolacao_lagrange(pontos, x) for x in valores_x]

    figura, eixo = plt.subplots(figsize=(9, 6), facecolor="#f5f6f8")
    eixo.set_facecolor("#f5f6f8")
    eixo.plot(valores_x, valores_y, color="#3567ff", linewidth=2, label="Polinomio P(x)")
    eixo.scatter(
        [x for x, _ in pontos],
        [y for _, y in pontos],
        color="#e67900",
        edgecolors="white",
        linewidths=0.8,
        s=55,
        zorder=3,
        label="Pontos de interpolacao",
    )

    for x, y in pontos:
        eixo.annotate(
            f"({x:g}, {y:g})",
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
            color="#344054",
        )

    eixo.set_title(f"Interpolacao de Lagrange {_formatar_polinomio(coeficientes)}")
    eixo.set_xlabel("Coordenada x")
    eixo.set_ylabel("Coordenada y")
    eixo.grid(True, color="#d7dce2", linewidth=0.8)
    eixo.axhline(0, color="#98a2b3", linewidth=0.8)
    eixo.axvline(0, color="#98a2b3", linewidth=0.8)
    eixo.legend()
    figura.tight_layout()
    plt.show()


def ler_ponto(indice: int, quantidade: int) -> tuple[float, float]:
    while True:
        texto = input(
            f"Ponto {indice} de {quantidade} (digite x,y; exemplo: 0,1): "
        ).strip()
        if texto.startswith("(") and texto.endswith(")"):
            texto = texto[1:-1]

        coordenadas = texto.split(",")
        if len(coordenadas) != 2:
            print(
                "Formato invalido. Digite x e y separados por virgula, como 0,1."
            )
            continue

        try:
            return float(coordenadas[0].strip()), float(coordenadas[1].strip())
        except ValueError:
            print("Coordenadas invalidas. Use numeros, por exemplo: 0,1 ou (0,1).")


def main() -> None:
    print("Interpolacao polinomial pelo metodo de Lagrange")

    try:
        quantidade = int(input("Quantos pontos voce quer usar? "))
        if quantidade <= 0:
            raise ValueError("A quantidade de pontos deve ser positiva.")

        print(f"Digite {quantidade} ponto(s), um por linha.")
        print("Em cada linha, informe x,y. Exemplo: 0,1 (use ponto para decimais).")
        pontos: list[tuple[float, float]] = []
        for indice in range(quantidade):
            pontos.append(ler_ponto(indice + 1, quantidade))

        x_calculo = float(input("Informe o valor de x para interpolar: "))
        valor = interpolacao_lagrange(pontos, x_calculo)
    except (ValueError, OverflowError) as erro:
        print(f"Nao foi possivel calcular: {erro}")
        return

    print(f"Valor interpolado em x = {x_calculo:g}: y = {valor:.6f}")
    try:
        exibir_grafico_lagrange(pontos)
    except (ImportError, OverflowError, ValueError) as erro:
        print(f"Nao foi possivel exibir o grafico: {erro}")


if __name__ == "__main__":
    main()
