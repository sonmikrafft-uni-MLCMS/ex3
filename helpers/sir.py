# module implementing the SIR model


def mu(b: float, I: float, mu0: float, mu1: float) -> float:
    """Calculate the per capita recovery rate of infectious individuals.

    :param b: number of available beds per 10k persons
    :type b: float
    :param I: number of infective persons
    :type I: float
    :param mu0: minimum recovery rate based on the number of available beds
    :type mu0: float
    :param mu1: maximum recovery rate based on the number of available beds
    :type mu1: float
    :return: per capita recovery rate
    :rtype: float
    """
    mu = mu0 + (mu1 - mu0) * (b / (I + b))
    return mu


def R0(beta: float, d: float, nu: float, mu1: float) -> float:
    """Calculate the basic reproduction number.
    Is expected number of cases directly generated by one case in a population where all individuals are susceptible.

    :param beta: average number of adequate contacts per unit time with infectious individuals
    :type beta: float
    :param d: per capita natural death rate
    :type d: float
    :param nu: per capita disease-induced death rate
    :type nu: float
    :param mu1: maximum recovery rate based on the number of available beds
    :type mu1: float
    :return: basic reproduction number
    :rtype: float
    """
    return beta / (d + nu + mu1)


def h(
    I: float,
    mu0: float,
    mu1: float,
    beta: float,
    A: float,
    d: float,
    nu: float,
    b: float,
) -> float:
    """Indicator function for bifurcations.

    computes value of indicator function h(I) for a given SIR model and value I

    :param I: number of infective persons
    :type I: float
    :param mu0: minimum recovery rate based on the number of available beds
    :type mu0: float
    :param mu1: maximum recovery rate based on the number of available beds
    :type mu1: float
    :param beta: average number of adequate contacts per unit time with infectious individuals
    :type beta: float
    :param A: recruitment or birth rate of susceptible population
    :type A: float
    :param d: per capita natural death rate
    :type d: float
    :param nu: per capita disease-induced death rate
    :type nu: float
    :param b: number of available beds per 10k persons
    :type b: float
    :return: value for indicator function h(I)
    :rtype: float
    """
    c0 = b ** 2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    res = c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3
    return res


def model(
    t: float,
    y: tuple[float, float, float],
    mu0: float,
    mu1: float,
    beta: float,
    A: float,
    d: float,
    nu: float,
    b: float,
) -> tuple[float, float, float]:
    """SIR model including hospitalization and natural death.

    :param t: time value for differential equation
    :type t: float
    :param y: tuple of S, I, R to pass current state values
    :type y: tuple[float, float, float]
    :param mu0: minimum recovery rate based on the number of available beds
    :type mu0: float
    :param mu1: maximum recovery rate based on the number of available beds
    :type mu1: float
    :param beta: average number of adequate contacts per unit time with infectious individuals
    :type beta: float
    :param A: recruitment or birth rate of susceptible population
    :type A: float
    :param d: per capita natural death rate
    :type d: float
    :param nu: per capita disease-induced death rate
    :type nu: float
    :param b: number of available beds per 10k persons
    :type b: float
    :return: tuple of partial derivatives of state S, I, R w.r.t time
    :rtype: tuple[float, float, float]
    """
    S, I, R = y[:]

    m = mu(b, I, mu0, mu1)

    dSdt = A - d * S - (beta * S * I) / (S + I + R)
    dIdt = -1 * (d + nu) * I - m * I + (beta * S * I) / (S + I + R)
    dRdt = m * I - d * R

    return (dSdt, dIdt, dRdt)
