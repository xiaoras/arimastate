import numpy as np
import sympy as sym
from scipy import linalg
from scipy.optimize import minimize

B = sym.Symbol('B') # define the symbol "B" (backshift operator), to be used in the polynomials

class Variable():

  def __init__(self, model, name, ARIMA, boxcox, input_s):
    
    self.model = model
    self.name = name
    self.boxcox = boxcox # the lambda-parameter of Box-Cox transformation

    # ARIMA noise
    self.sigma = ARIMA[0]
    self.AR = sym.poly(ARIMA[1], B)
    self.MA = sym.poly(ARIMA[2], B)

    # stochastic input
    AR_s, MA_s = {}, {}
    for key, value in input_s.items():
      AR_s[key] = sym.poly(value[0], B)
      MA_s[key] = sym.poly(value[1], B)
    self.AR_s = AR_s # dict associating the name of the input variable with the AR part of the transfer function
    self.MA_s = MA_s # dict associating the name of the input variable with the MA part of the transfer function
    self.input_s = input_s # list of names of input stochastic variables

  def construct(self): # generate self.Psi, self.F and self.G

    p, q = sym.degree(self.AR, B), sym.degree(self.MA, B)
    m = max(p, q)
    Psi = self._find_Psi(self.AR, self.MA, p, q, m)
    F = self._find_F(self.AR, m)
    G = self._find_G(self, m)

    for y in self.model.v: # loop over model's variables

      if self.name in y.input_s: # if Variable is input of y, then proceed
      
        AR_y = y.AR_s[self.name] * self.AR
        MA_y = y.MA_s[self.name] * self.MA

        p_y, q_y = sym.degree(AR_y, B), sym.degree(MA_y, B)
        m_y = max(p_y, q_y)

        Psi_y = self._find_Psi(AR_y, MA_y, p_y, q_y, m_y)
        Psi = np.concatenate((Psi, Psi_y))
        
        F_y = self._find_F(AR_y, m_y)
        F = linalg.block_diag(F, F_y)
        
        G_y = self._find_G(y, m_y)
        G = np.concatenate((G, G_y), axis=1)
    
    self.Psi = Psi
    self.F = F
    self.G = G

  def _find_Psi(self, AR, MA, p, q, m): # generate column-vector Psi from AR and MA polynomials

    phi = -np.flip(sym.poly(AR, B).all_coeffs())
    the = -np.flip(sym.poly(MA, B).all_coeffs())
    phi = np.concatenate((phi, np.zeros(m-p)))
    the = np.concatenate((the, np.zeros(m-q)))

    psi = []
    for j in range(m+1):
      psi_j = (the[j] - phi[1:j+1] @ np.flip(psi[:j]))/phi[0]
      psi.append(psi_j)

    return np.array([psi]).T

  def _find_F(self, AR, m): # generate matrix F from the AR polynoimal and m = max(p, q)

    zero_vec = np.zeros((m+1, 1)) # column-vector of m+1 zeros
    
    I = np.identity(m) # mxm identity matrix

    phi_vec = sym.poly(AR, B).all_coeffs()[:-1] # list of AR coefficients [phi_{p}, phi_{p-1}, .., phi_{1}]
    phi_vec = (m - len(phi_vec))*[0] + phi_vec # append 0's in front, so that len(phi_vec) = m
    phi_vec = np.array([-phi for phi in phi_vec]) # turn phi_vec into a row-vector (array) of floats

    F = np.hstack((zero_vec, np.vstack((I, phi_vec)))) # stack: zero_vec | (I / phi_vec)

    return F

  def _find_G(self, x, m): # generate matrix G for variable x with m = max(p, q)
    
    n = len(self.model.v)
    G = np.zeros((n,m+1))

    i = np.where(np.array(self.model.v) == x) # get the index of variable x in model.v
    G[i,0] = 1 # change 0 to 1 for the element in the first column and ith
    
    return G

class Model():

  def __init__(self):

    self.v = [] # variables

  def parameters(self, param_number):
    
    p = []
    for i in range(param_number):
      char = 'p{}'.format(i)
      p.append(sym.Symbol(char))
    
    self.p = p # list of parameter symbols

  def variable(self, name, ARIMA, boxcox=(1,1), input_s={}): # add a stochastic variable to the model
    
    x = Variable(self, name, ARIMA, boxcox, input_s)
    self.v.append(x)

  def evaluate(self, *values): # call this method after adding all the model's variables
  # generate self.G, self.F, self.Psi and self.Q based on the parameter values in the list "values"

    # create the symbol -> value replacer

    r = [(symbol, value) for symbol, value in zip(self.p, values)]

    # make a temporary model, identical to self, but with parameter values instead of symbols

    m_temp = Model()
    m_temp.parameters(6)
    for x in self.v:
      input_s = {}
      for y_name in x.input_s:
        input_s[y_name] = [x.AR_s[y_name].subs(r), x.MA_s[y_name].subs(r)]

      m_temp.variable(x.name, [x.sigma, x.AR.subs(r), x.MA.subs(r)], x.boxcox, input_s)

    # construct F, Psi, G and Q based on the evaluated variables

    F, Psi, G = [], [], []
    for x in m_temp.v:
      x.construct()
      G.append(x.G)
      F.append(x.F)
      Psi.append(x.Psi)

    self.G = np.array(np.concatenate(G, axis=1)).astype(float)
    self.F = np.array(linalg.block_diag(*F)).astype(float)
    self.Psi = np.array(linalg.block_diag(*Psi)).astype(float)

    sigmas = np.array([[x.sigma for x in m_temp.v]])
    W = self.Psi * sigmas
    self.Q = W @ W.T

  def simulate(self, X0, T): # can be called after the model has been evaluated
  # simulate the model by solving the state equation up to time T, starting from initial condition X0
  # input X0 must be a column vector with length equal the size of F matrix
  # the output Y is a T x n matrix, where n is the number of variables in the model

    # check on correct input size
    if X0.shape[0] != self.F.shape[1]:
      print("Error: X0 must be a column vector with {} rows!".format(self.F.shape[1]))
      return

    # white noises mean and covariance
    Mu = np.zeros(len(self.v))
    sigma2s = [x.sigma**2 for x in self.v]
    Sigma2 = linalg.block_diag(*sigma2s)

    # state equation
    X = [X0]
    for t in range(1, T):
      w = np.random.multivariate_normal(Mu, Sigma2, (1)).T
      X_new = self.F @ X[-1] + self.Psi @ w
      X.append(X_new)
    X = np.array(X)
    Y = np.einsum("ij,ljk->li", self.G, X)

    return self._boxcox(Y, inverse=True)

  def forecast(self, X0, P0, Y): # can be called after the model has been evaluated
  # forecast via Kalman equations, based on observations Y and initial conditions X0 and P0
  # inputs X0 and P0 are respectively a column vector with length equal the size of F matrix, and its covariance matrix
  # input Y is a T x n matrix, where n is the number of variables and T the timespan covered
  # the output Y_hat is a T x n matrix, to be compared with Y
  # the outputs epsilon and sigma are respectively T x n and T x n x n, the vector of innovations and its covariance matrix

    # check on correct input size
    if X0.shape[0] != self.F.shape[1] or P0.shape != self.F.shape:
      print("Error: X0 and P0 must be a column vector and a square matrix with size {}!".format(self.F.shape[1]))
      return
    if Y.shape[1] != len(self.v):
      print("Error: Y must be a T x {} matrix".format(len(self.v)))
      return

    Y = self._boxcox(Y, inverse=False) # apply boxcox transformation
    Y = np.expand_dims(Y.T, 1) # turn Y into n x 1 x T tensor: for each t, we thus have a a column-vector (i.e., n x 1 matrix)

    F, Q, G = self.F, self.Q, self.G

    X, P = [X0], [P0]
    Y_hat, Sigma_hat = [], []
    for t in range(Y.shape[2]):
      
      K = P[-1] @ G.T @ linalg.inv(G @ P[-1] @ G.T)

      X_t = F @ (X[-1] + K @ (Y[:,:,t] - G @ X[-1]))
      P_t = F @ (P[-1] - K @ G @ P[-1]) @ F.T + Q

      X.append(X_t)
      P.append(P_t)

      Y_t = G @ X[-1] # Y_{t+1,t}
      Sigma_t = G @ P[-1] @ G.T # covariance
      Y_hat.append(Y_t)
      Sigma_hat.append(Sigma_t)
    
    Y_hat = np.einsum("ijk->ij", np.array(Y_hat))
    Y = np.einsum("jki->ij", np.array(Y))

    Y_hat = np.insert(Y_hat[:-1], 0, Y[0], axis=0)

    # errors (i.e., innovations) and their covariance matrix
    epsilon = Y - Y_hat
    sigma = np.array(Sigma_hat)

    return self._boxcox(Y_hat, inverse=True), epsilon, sigma

  def logL(self, parameters, X0, P0, Y):
  # (negative) log-likelihood for parameters, given initial data X0 and P0, and observations Y

    self.evaluate(*parameters)

    _, epsilon, sigma = self.forecast(X0, P0, Y)
    
    S = []
    for t in range(epsilon.shape[0]):
      e, s = epsilon[t], sigma[t]
      S_t = (np.log(abs(linalg.det(s))) + e.T @ linalg.inv(s) @ e)/2
      S.append(S_t)
    S = float(sum(S))

    return S

  def MLE(self, params, X0, P0, Y, method='L-BFGS-B'):
  # Maximum Likelihood Estimation (method used in scipy.optimize.minimize)
 
    res = minimize(self.logL, params, args=(X0, P0, Y), method=method)

    self.estimated_p = res.x
    self.evaluate(*self.estimated_p)

  def _boxcox(self, Y, inverse=False):
    
    Z = np.copy(Y)

    if inverse: # apply the inverse transformation
      for x, i in zip(self.v, range(Z.shape[1])):
        l1, l2 = x.boxcox
        if l1 == 0:
          Z[:,i] = np.exp(Y[:,i]) - l2
        else:
          Z[:,i] = (l1*Y[:,i] + 1)**(1/l1) - l2
    
    else: # apply the direct transformation
      for x, i in zip(self.v, range(Z.shape[1])):
        l1, l2 = x.boxcox
        if l1 == 0:
          Z[:,i] = np.log(Y[:,i] + l2)
        else:
          Z[:,i] = ((Y[:,i] + l2)**l1 - 1)/l1

    return Z