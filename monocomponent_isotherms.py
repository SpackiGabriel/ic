import numpy as np


#################### Langmuir ####################
def langmuir(p, qe, param, relative=False):

    # Inicializa o resultado acumulado como 0.
    result = 0

    # Itera por todos os índices de 'p' (pressões fornecidas)
    for i in range(len(p)):
        try:
            # Ignora warnings de erro do NumPy, como divisão por zero.
            np.seterr(all='ignore')
            
            # Extrai os parâmetros do modelo Langmuir.
            q_max = param[0] # Capacidade máxima de adsorção.
            b = param[1] # Constante de afinidade.

            # Calcula o numerador da equação de Langmuir: 'q_max * b * p[i]'.
            over = q_max * b * p[i]

            # Calcula o denominador da equação: '1 + b * p[i]'.
            under = 1 + b * p[i]

            # Calcula o valor predito da adsorção com a equação de Langmuir.
            value = over / under
            
            # Se o parâmetro 'relative' estiver ativado, calcula o erro relativo quadrado.
            if relative:
                result = result + abs((qe[i] - value) / qe[i]) ** 2
            else:
                # Caso contrário, calcula o erro absoluto quadrado.
                result = result + abs(qe[i] - value) ** 2
        
        # Captura erros potenciais (ex.: valores fora do intervalo ou divisão por zero).
        except ValueError or TypeError or IndexError or ZeroDivisionError:

            # Adiciona um grande valor ao resultado para penalizar a solução inválida.
            result = result + 1e8

    # Retorna o erro acumulado (ou soma dos erros quadrados) como resultado.
    return result


def __langmuir_q__(press, param):
    q_max = param[0]
    b = param[1]
    return q_max * b * press / (1 + b * press)


#################### Sips ####################
def sips(p, qe, param, relative=False):
    result = 0
    for i in range(len(p)):
        try:
            np.seterr(all='ignore')
            qmax = param[0]
            b = param[1]
            nS = param[2]
            over = param[0] * (b * p[i]) ** (1 / nS)
            under = 1 + (b * p[i]) ** (1 / nS)
            value = over / under
            if relative:
                result = result + abs((qe[i] - value) / qe[i]) ** 2
            else:
                result = result + abs(qe[i] - value) ** 2
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            result = result + 1e8
    return result


def __sips_q__(press, param):
    q_max = param[0]
    b = param[1]
    nS = param[2]
    return (q_max * (b * press) ** (1 / nS)) / (1 + (b * press) ** (1 / nS))


#################### Toth ####################
def toth(p, qe, param, relative=False):
    result = 0
    for i in range(len(p)):
        try:
            np.seterr(all='ignore')
            q_max = param[0]
            b = param[1]
            n = param[2]
            over = q_max * b * p[i]
            under = (1 + (b * p[i]) ** n) ** (1 / n)
            value = over / under
            if relative:
                result = result + abs((qe[i] - value) / qe[i]) ** 2
            else:
                result = result + abs(qe[i] - value) ** 2
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            result = result + 1e8
    return result


def __toth_q__(p, param):
    q_max = param[0]
    b = param[1]
    n = param[2]
    over = q_max * b * p
    under = (1 + (b * p) ** n) ** (1 / n)
    result = over / under
    return result


#################### BET ####################
def bet(p, qe, param, relative=False):
    result = 0
    for i in range(len(p)):
        try:
            qm = param[0]
            C = param[1]
            np.seterr(all='ignore')
            over = qm * C * p[i]
            under = (1 - p[i]) * (1 - p[i] + C * p[i])
            value = over / under
            if relative:
                result = result + abs((qe[i] - value) / qe[i]) ** 2
            else:
                result = result + abs(qe[i] - value) ** 2
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            result = result + 1e8
    return result


def __bet_q__(p, param):
    qm = param[0]
    C = param[1]
    over = qm * C * p
    under = (1 - p) * (1 - p + C * p)
    result = over / under
    return result


#################### gab ####################
def gab(p, qe, param, relative=False):
    result = 0
    for i in range(len(p)):
        try:
            qm = param[0]
            C = param[1]
            K = param[2]
            np.seterr(all='ignore')
            over = qm * C * K * p[i]
            under = (1 - K * p[i]) * (1 - K * p[i] + C * K * p[i])
            value = over / under
            if relative:
                result = result + abs((qe[i] - value) / qe[i]) ** 2
            else:
                result = result + abs(qe[i] - value) ** 2
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            result = result + 1e8
    return result


def __gab_q__(p, param):
    qm = param[0]
    C = param[1]
    K = param[2]
    np.seterr(all='ignore')
    over = qm * C * K * p
    under = (1 - K * p) * (1 - K * p + C * K * p)
    result = over / under
    return result


#################### Multisite Langmuir ####################
def f_langmuir_multi(p, q, param):
    q_max = param[0]
    b = param[1]
    a = param[2]
    return q - q_max * b * p * ((1 - (q / q_max)) ** a)


def f_prime_langmuir(p, q, param):
    q_max = param[0]
    b = param[1]
    a = param[2]

    return 1 + b * p * ((1 - q / q_max) ** a) * a / (1 - q / q_max)


def newton(p, q, param):
    for i in range(15):
        try:
            newq = q - f_langmuir_multi(p, q, param) / f_prime_langmuir(p, q, param)
            if abs(q - newq) < 1e-5:
                return newq
            q = newq
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            return 1e8
    return q


def multisite_langmuir(p, qe, param, relative=False):
    result = 0
    for i in range(len(p)):
        try:
            np.seterr(all='ignore')
            q = 0.5
            if relative:
                result = result + abs(qe[i] - newton(p[i], q, param) / qe[i]) ** 2
            else:
                result = result + abs(qe[i] - newton(p[i], q, param)) ** 2
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            result += 1e8

    return result


def __langmuir_multi_q__(p, param):
    q = 0.5
    for i in range(20):
        try:
            newq = q - f_langmuir_multi(p, q, param) / f_prime_langmuir(p, q, param)
            if abs(q - newq) < 1e-5:
                return newq
            q = newq
        except TypeError or ValueError:
            return 0
    return q


#################### Modified BET####################

def modified_bet(p, qe, param, relative=False):
    result = 0
    for i in range(len(p)):
        try:
            qm = param[0]
            C = param[1]
            m = param[2]
            np.seterr(all='ignore')
            over = qm * C * p[i]
            under = ((1 - p[i])**m) * (1 + C * p[i])
            value = over / under
            if relative:
                result = result + abs((qe[i] - value) / qe[i]) ** 2
            else:
                result = result + abs(qe[i] - value) ** 2
        except ValueError or TypeError or IndexError or ZeroDivisionError:
            result = result + 1e8
    return result


def __modified_bet_q__(p, param):
    qm = param[0]
    C = param[1]
    m = param[2]
    over = qm * C * p
    under = ((1 - p)**m) * (1 + C * p)
    result = over / under
    return result
