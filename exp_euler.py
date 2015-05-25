#!/usr/bin/env python
#############################################################################
# course:   Numerische Methoden D-PHYS
# exercise: assignment 10
# author:   Thomas Diggelmann <thomas.diggelmann@student.ethz.ch>
# date:     25.04.2015
#############################################################################
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
    one = eye(y.shape[1])
    for k in xrange(nsteps-1):
        z = h*Df(y[k,:])
        phi = (expm(z)-one).dot(inv(z))
        y[k+1,:] = y[k,:] + h*phi.dot(f(y[k,:]))

    return ts, y


if __name__ == '__main__':

    #########################################
    #                                       #
    # Definieren Sie hier die Jacobi-Matrix #
    #                                       #
    #########################################

    Df = lambda y: array([[-((2*y[0])/y[1]), 1 + y[0]**2/y[1]**2 + log(y[1])],
                          [-1, 0.0]])

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
    title("Exponentielles Euler-Verfahren")
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
    errors_end = []  
    errors_max = []  

    ########################################
    #                                      #
    # Erstellen Sie hier einen loglog-Plot #
    #                                      #
    ########################################
    for nsteps in N:
        ts, y = expEV(nsteps, to, te, y0, f, Df)
        h = (te-to)/(nsteps-1)

        err_max = max(norm(y-sol(ts), axis=1))
        err_end = norm(y[-1]-sol(te))
    
        hs.append(h)
        errors_end.append(err_end)
        errors_max.append(err_max)

    hs = array(hs)
    errors_max = array(errors_max)
    errors_end = array(errors_end)

    title("Exponentielles Euler-Verfahren Konvergenz")
    loglog(hs, errors_max, label=r"$\epsilon_{max}:=max||y_{exp}(t)-y_{sol}(t)||_2 \forall t$")
    loglog(hs, errors_end, label=r"$\epsilon_{end}:=||y_{exp}(t_{end})-y_{sol}(t_{end})||_2$")
    legend(loc="lower right")
    xlabel('$h$')
    ylabel('Abs. Fehler')
    savefig('exp_euler_konvergenz.pdf')
    show()

    print("Konvergenzrate fuer Exp. Euler mit maximalem Fehler: %.4f" % polyfit(log(hs), log(errors_max), 1)[0])
    print("Konvergenzrate fuer Exp. Euler mit Fehler zum Endzeitpunkt: %.4f" % polyfit(log(hs), log(errors_end), 1)[0])