def finite_difference_derivative(
    coefficients: list, x: float, h: float
) -> tuple[float, float, float]:

    def f(x):
        result = 0.0
        for i, coefficient in enumerate(coefficients):
            result += coefficient * (x ** i)
        return result

    f_x = f(x)
    f_x_h = f(x + h)

    slope = (f_x_h - f_x) / h

    return f_x, f_x_h, slope