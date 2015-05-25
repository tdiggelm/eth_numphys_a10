#!/usr/bin/env python
from numpy import *
from matplotlib.pyplot import *
from scipy.linalg import expm, solve, inv
from numpy.linalg import norm

def expEV(nsteps, to, te, y0, f, Df):
    """
    Exponentielles Eulerverfahren

    Keyword Arguments:
    nsteps -- Anzahl Zeitschritte
    y0     -- Startwert zur Zeit t=to
    to     -- Anfangszeit
    te     -- Endzeit
    f      -- y'
    Df     -- Jacobi-Matrix

    Return Arguments:
    ts     -- [to ... te]
    y      -- Loesung y
    """

    ts, h = linspace(to, te, nsteps, retstep=True)

    # Speicherallokation
    y = zeros((nsteps, 2))

    # Startwert
    y[0,:] = y0

    ############################################################
    #                                                          #
    # Implementieren Sie hier das exponentielle Eulerverfahren #
    #                                                          #
    ############################################################

    return ts, y


if __name__ == '__main__':


    #########################################
    #                                       #
    # Definieren Sie hier die Jacobi-Matrix #
    #                                       #
    #########################################
    Df = lambda y: array([[0.0, 0.0],
                          [0.0, 0.0]])

    # Rechte Seite
    f = lambda y: array([-y[0]**2/y[1] + y[1]*log(y[1]),
                         -y[0]])

    # Exakte Loesung
    sol = lambda t: array([-cos(t)*exp(sin(t)), exp(sin(t))]).T

    # Anfangswert
    y0 = array([-1.0, 1.0])

    to = 0.0
    te = 6.0
    nsteps = 20
    ts, y = expEV(nsteps, to, te, y0, f, Df)

    t_ex = linspace(to, te, 1000)
    y_ex = sol(t_ex)

    figure()
    plot(ts, y[:,0], 'r-x', label=r'$y[0]$')
    plot(ts, y[:,1], 'g-x', label=r'$y[1]$')
    plot(t_ex, y_ex[:,0],'r', label=r'$y_{ex}[0$]')
    plot(t_ex, y_ex[:,1],'g', label=r'$y_{ex}[1$]')
    legend(loc='best')
    xlabel('$t$')
    ylabel('$y$')
    grid(True)
    savefig('exp_euler.pdf')
    show()

    figure()
    N = [24, 48, 96, 192, 384]
    hs = []
    errors = []

    ########################################
    #                                      #
    # Erstellen Sie hier einen loglog-Plot #
    #                                      #
    ########################################
    xlabel('$h$')
    ylabel('Abs. Fehler')


    savefig('exp_euler_konvergenz.pdf')
    show()

    ############################################
    #                                          #
    # Berechnen Sie hier die Konvergenzordnung #
    #                                          #
    ############################################
    conv_rate = nan

    print 'Exponentielles Eulerverfahren konvergiert mit algebraischer Konvergenzordnung: %.2f' % conv_rate
