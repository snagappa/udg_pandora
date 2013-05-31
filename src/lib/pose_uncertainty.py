#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from numpy.linalg import norm
from scipy.linalg import block_diag
import scipy.optimize
import scipy.io

import code

def compute_pose_unc(image_points, mosaic_points, camera_matrix, wNm,
    pixel_noise_var, homography_matrix, camera_cov=None, wNm_cov=None, h=1e-6):
    """compute_pose_unc(image_points, mosaic_points, camera_matrix, wNm,
                        pixel_noise_var, homography_matrix, camera_cov=None,
                        wNm_cov=None, h=1e-6)
    where
        image_points -> Nx2 set of points in the camera image
        mosaic_points -> Nx2 set of corresponding points in the mosaic/template
        camera_matrix -> 3x3 camera matrix
        wNm -> Homography from world to mosaic/template (?)
        pixel_noise_var -> Noise variance for the pixel positions
        homography_matrix -> Initial homography from mosaic/template to image
        camera_cov -> 4x4 matrix for camera matrix uncertainty (default eye(4))
        wNm_cov -> 9x9 matrix of the covariance of wNm (default zeros((9, 9)))
        h -> Step size for optimisation
    """
    import pose_uncertainty
    import numpy as np
    if camera_cov is None:
        camera_cov = np.eye(4)
    
    if wNm_cov is None:
        wNm_cov = np.zeros((9, 9))
    
    # Normalise(?) camera matrix
    camera_matrix /= camera_matrix[2, 2]
    # Normalisation of camera matrix covariance (use Frobenius norm)
    kscale = 1./norm(camera_matrix) 
    kvec_scaled = camera_matrix[[0, 1, 0, 1], [0, 1, 2, 2]] * kscale
    
    assert np.abs(norm(wNm) - 1) < 1e-6, "Norm of wNm must be 1"
    
    # Initial homography matrix
    # where image_points = homography_matrix * mosaic_points
    Mimini, _CONDS, _A, _R = lsnormplanar(image_points, mosaic_points, 'stdv')
    MiWini = np.dot(Mimini, np.linalg.inv(wNm))
    
    # Generate initial pose parameters
    poseparams_ini = getposeparam_sturm(camera_matrix, MiWini, "WTC")
    
    num_points = image_points.shape[0]
    x_coord = (image_points*kscale).flatten()
    var_x_coord = (pixel_noise_var*kscale**2)*np.ones(num_points*2)
    
    XL = np.hstack((kvec_scaled, wNm.flatten(), x_coord.flatten()))
    
    fcost_ini = function_cost_total(poseparams_ini, XL, kscale, mosaic_points)
    print 'Initial cost: ', fcost_ini
    
    # Refine the pose using Simplex Downhill
    #Tfin, fopt, num_iters, num_funcalls, warnflag = scipy.optimize.fmin(
    #    function_cost_total, poseparams_ini, (XL, kscale, mosaic_points),
    #    xtol=1e-6, ftol=1e-6, full_output=1)
    result = scipy.optimize.minimize(function_cost_total, poseparams_ini,
                                     (XL, kscale, mosaic_points),
                                     method="L-BFGS-B") #"Nelder-Mead")
    if not result.success:
        print "Minimization did not converge!"
    Tfin = result.x
    fcost_fin = result.fun #function_cost_total(Tfin, XL, kscale, mosaic_points)
    print 'Initial cost: ', fcost_ini
    print 'Final cost:   ', fcost_fin
    
    Theta = Tfin
    POSEPAR = Tfin	# Devolver um vector linha de par�metros
    
    camera_cov_scaled = camera_cov * kscale**2
    # S� se est� a considerar ruido independente nas coordenadas
    cov_x_coor = np.diag(var_x_coord)
    # Adicionar covari�ncia dos par�metros intr�nsecos
    cov_XL = block_diag(camera_cov_scaled, wNm_cov, cov_x_coor)
    
    # Calculo da Hessiana de F em ordem a Tetha
    d2FdTheta2 = function_d2FdTheta2(Theta, XL, kscale, mosaic_points, h)
    
    # ==== Checked up to here ====
    
    # Calculo da segunda derivada de F em ordem a Tetha e PSI
    log_d2FdThetadXL = np.empty((XL.shape[0], Theta.shape[0]))
    sign_d2FdThetadXL = np.empty((XL.shape[0], Theta.shape[0]))
    inv_hsq = 1./(h**2)
    log = np.log
    sign = np.sign
    log_inv_hsq = log(inv_hsq)
    half_h = h/2.
    
    print "compute_pose_unc():100"
    code.interact(local=locals())
    
    for n in range(XL.shape[0]): #1:length(XL)
        for p in range(Theta.shape[0]): #1:length(Theta)
            #pose_vecs = np.asarray((vadd(Theta, p,  half_h), vadd(Theta, p, -half_h), vadd(Theta, p,  half_h), vadd(Theta, p, -half_h)))
            #xl_vecs = np.asarray((vadd(XL, n,  half_h), vadd(XL, n,  half_h), vadd(XL, n, -half_h), vadd(XL, n, -half_h)))
            P1 = function_cost_total(vadd(Theta, p,  half_h), vadd(XL, n,  half_h), kscale, mosaic_points)
            P2 = function_cost_total(vadd(Theta, p, -half_h), vadd(XL, n,  half_h), kscale, mosaic_points)
            P3 = function_cost_total(vadd(Theta, p,  half_h), vadd(XL, n, -half_h), kscale, mosaic_points)
            P4 = function_cost_total(vadd(Theta, p, -half_h), vadd(XL, n, -half_h), kscale, mosaic_points)
            #% troca de n e p em relacao a numxdiff.m
            p1_p4 = P1 + P4
            p2_p3 = P2 + P3
            P_sum = (P1 + P4) -(P2 + P3)
            P_sign = sign(P_sum)
            log_d2FdThetadXL[n, p] = log_inv_hsq + log(abs(P_sum))
            sign_d2FdThetadXL[n, p] = sign(P_sign)
    
    d2FdThetadXL = sign_d2FdThetadXL*np.exp(log_d2FdThetadXL)
    
    print "compute_pose_unc():123"
    code.interact(local=locals())

    # Para debug: Calculo do Jacobiano da funcao que calcula explicitamente o 
    # Theta (getposeparam_sturm)
    #inv_h = 1./h
    #dThetadXL = np.zeros((XL.shape[0], Theta.shape[0]))
    #for n in range(XL.shape[0]):
    #    Theta1 = function_pose_from_XL(vadd(XL, n,  half_h), kscale, mosaic_points)
    #    Theta2 = function_pose_from_XL(vadd(XL, n, -half_h), kscale, mosaic_points)
    #    # troca de n e p em relacao a numxdiff.m
    #    dThetadXL[n, :] = inv_h * (Theta1 - Theta2)
    #
    #COVPOSE_dir = np.dot(dThetadXL.T, np.dot(cov_XL, dThetadXL))
    
    inv_d2FdTheta2 = np.linalg.inv(d2FdTheta2)
    COVPOSEPAR = np.dot(inv_d2FdTheta2, 
                        np.dot(d2FdThetadXL.T, 
                               np.dot(cov_XL, np.dot(d2FdThetadXL, 
                                                     inv_d2FdTheta2))))
    print "compute_pose_unc():143"
    code.interact(local=locals())
    
    return POSEPAR, COVPOSEPAR#, COVPOSE_dir
    
    
    
    
    
def function_cost_total(Thetain, XLin, kscale, COORm):
    """funcao de custo usando os parametros intrinsecos e a colineacao que 
    calibra metricamente o mosaico"""
    K = np.asarray([[XLin[0], 0, XLin[2]],
                    [0, XLin[1], XLin[3]],
                    [0, 0, kscale]])
    wNm = XLin[4:13].reshape((3, 3))
    TiW = pose1homo(Thetain, K, 'WTC')
    
    PSI = np.dot(TiW, wNm)
    PSI /= norm(PSI, 'fro')
    
    image_points = XLin[13:].copy()
    num_points = image_points.shape[0]/2
    image_points = image_points.reshape((num_points, 2))/kscale
    
    COORm_em_i = trancoor(PSI, COORm)
    fcost = norm(image_points - COORm_em_i, 'fro')/1000.
    
    return fcost


def pose1homo(poseparams, K, wtcflag="CTW"):
    """TiW = pose1homo(POSEPAR,K)
    Homography from Pose Parameters
    
    Cria uma homografia TiW a partir de par�metros de pose.
    Usa os seguintes par�metros de pose POSEPAR = [a b c tx ty tz].
    Os �ngulos a, b e c s�o os �ngulos fixos X-Y-Z tal como definidos no
    Craig 2� Ed. Pag 46, para a rota��o cRw. Os elementos tx, ty e tz
    correspondem � transla��o do ref da camera cTw.
    
    TiW = pose1homo(POSEPAR,K,'WTC')
    Usa os par�metros de pose na forma POSEPAR = [a b c wtcx wtcy wtcz], na qual os elementos wtcx, 
    wtcy e wtcz correspondem � transla��o do ref da camera wTc.
    """
    
    wtcflag = wtcflag.upper()
    assert wtcflag in ["WTC", "CTW"], "wtcflag must be 'CTW' or 'WTC'"
    a, b, c = poseparams[:3]
    
    cos_a, cos_b, cos_c = np.cos((a, b, c))
    sin_a, sin_b, sin_c = np.sin((a, b, c))
    log_cos_a, log_cos_b, log_cos_c = np.log(np.cos((a, b, c)))
    log_sin_a, log_sin_b, log_sin_c = np.log(np.sin((a, b, c)))
    
    phi_1 = np.asarray((cos_a*cos_b, sin_a*cos_b, -sin_b))
    phi_2 = np.asarray((cos_a*sin_b*sin_c-sin_a*cos_c, 
                        sin_a*sin_b*sin_c+cos_a*cos_c, cos_b*sin_c))

    phi_1 = np.asarray((cos_a*cos_b, sin_a*cos_b, -sin_b))
    phi_2 = np.asarray((cos_a*sin_b*sin_c-sin_a*cos_c, 
                        sin_a*sin_b*sin_c+cos_a*cos_c, cos_b*sin_c))
    
    if wtcflag == 'CTW':
        ctw = poseparams[3:]
    else:
        wtc = poseparams[3:]
        phi_3 = np.asarray((cos_a*sin_b*cos_c+sin_a*sin_c,
                            sin_a*sin_b*cos_c-cos_a*sin_c, cos_b*cos_c))
        ctw = np.dot(-np.vstack((phi_1, phi_2, phi_3)).T, wtc)
    
    TiW = np.dot(K, np.vstack((phi_1, phi_2, ctw)).T)
    TiW /= TiW[2, 2]
    
    return TiW


def trancoor(transformation, coords):
    """TCOOR = trancoor(T, COOR)
    Apply Coordinate transformation:
    Aplica a transforma��o T(3x3) � lista COOR(nx2).
    """
    num_points = coords.shape[0]
    # Create homogeneous list
    coords = np.hstack((coords, np.ones((num_points, 1))))
    
    # Get transformed list
    tr_coords = np.dot(coords, transformation.T)
    
    # Divide by the homogeneous parameter
    tr_coords = tr_coords[:, :2]/tr_coords[:, np.newaxis, 2]
    return tr_coords


def vadd(vec_in, pos, amount):
    """vecout = vadd(VECIN, pos, amount)
    Add amount to vector VECIN in position pos
    """
    vec_out = vec_in.copy()
    vec_out[pos] = vec_in[pos] + amount
    return vec_out


def function_d2FdTheta2(Theta, XL, kscale, COORm, h):
    """
    d2FdTheta2 = function_d2FdTheta2(Theta, XL, kscale, COORm, h)
    Calculo da Hessiana de F em ordem a Tetha
    """
    num_theta = Theta.shape[0]
    half_h = h/2.
    inv_hsq = 1./(h**2)
    log = np.log
    sign = np.sign
    log_inv_hsq = log(inv_hsq)
    log_d2FdTheta2 = np.empty((num_theta, num_theta))
    sign_d2FdTheta2 = np.empty((num_theta, num_theta))
    for n in range(num_theta):
        for p in range(num_theta):
            P1 = function_cost_total(vadd(vadd(Theta, n, half_h), p, half_h), XL, kscale, COORm)
            P2 = function_cost_total(vadd(vadd(Theta, n, half_h), p, -half_h), XL, kscale, COORm)
            P3 = function_cost_total(vadd(vadd(Theta, n, -half_h), p, half_h), XL, kscale, COORm)
            P4 = function_cost_total(vadd(vadd(Theta, n, -half_h), p, -half_h), XL, kscale, COORm)
            # troca de n e p em relacao a numhess.m
            P_sum = P1 -P2 -P3 +P4
            P_sign = sign(P_sum)
            log_d2FdTheta2[n, p] = log_inv_hsq + log(abs(P_sum))
            sign_d2FdTheta2[n, p] = P_sign
    d2FdTheta2 = np.exp(log_d2FdTheta2)*sign_d2FdTheta2
    return d2FdTheta2


def function_pose_from_XL(XLin, kscale, mosaic_points):
    camera_matrix = np.asarray([[XLin[0], 0, XLin[2]],
                                [0, XLin[1], XLin[3]],
                                [0, 0, kscale]])
    wNm = np.reshape(XLin[4:13], (3, 3))
    
    image_points = XLin[13:].copy()
    num_points = image_points.shape[0]/2
    image_points = image_points.reshape((num_points, 2))/kscale
    
    PSImat, _CONDS, _A, _R = lsnormplanar(image_points, mosaic_points, 'stdv')
    Thetaout = getposeparam_sturm(camera_matrix,
                                  np.dot(PSImat, np.linalg.inv(wNm)), 'WTC')
    return Thetaout



def lsnormplanar(COOR1, COOR2, methodnorm):
    """M12 = lsnormplanar(COOR1, COOR2, methodnorm)
    Least Squares Normalized Planar Transformation
    
    Determina��o da Transforma��o Planar M12(3x3) que relaciona os pontos 
    com coordenadas COOR1 e COOR2. Esta fun��o normaliza as coordenadas 
    dos pontos impondo m�dia nula e permite ainda efectuar 
    escalamentos do tipo :
    methodnorm = 'nnor' : (no normalization). 
    methodnorm = 'none' : nenhum escalamento. 
    methodnorm = 'stdv' : desvio padr�o unit�rio em cada coordenada.
    methodnorm = 'sqrt' : desvio padr�o igual � raiz quadrada do d.p. original.
    S�o necess�rias pelo menos 4 pontos em cada lista de coordenadas.
    
    M12, CONDS = lsnormplanar(COOR1, COOR2) devolve tamb�m o n� de 
    condi��o do sistema (A'*A) referente 'as coordenadas normalizadas.
    M12, CONDS, A, R = lsnormplanar(COOR1, COOR2) devolve a matriz do 
    sistema A e o vector de res�duos R referente 'as coordenadas 
    normalizadas.
    
    Usa o sistema (u,v) para as coordenadas 2-D.
    
    M12 = lsplanar(COOR1, COOR2, methodnorm, methodplan)
    Permite a especifica��o de uma outra transforma��o planar:
      methodplan = 
          'lsplanar' : Transforma��o planar gen�rica.
    """
    
    assert COOR1.shape == COOR2.shape, (
        "Coordinate lists must have the same size.")
    
    # Normalize the coordinates
    NCL1, T1 = normcoor(COOR1, methodnorm)
    NCL2, T2 = normcoor(COOR2, methodnorm)
    
    # Call the usual planar transformation estimation function
    M12norm, CONDS, A, R = lsplanar(NCL1, NCL2, 'none')
    
    # Invert the effect of the normalization
    M12 = np.dot(np.linalg.inv(T1), np.dot(M12norm, T2))
    
    M12 /= M12[2, 2]
    #print "lsnormplanar():296"
    #code.interact(local=locals())
    
    return M12, CONDS, A, R


def normcoor(COOR, METHOD="none"):
    """NCOOR, T = normcoor(COOR)
    Normalize 2D Coordinates:
    
    Normaliza uma lista COOR de coordenadas 2D, tal que tenha m�dia nula.
    Devolve lista modificada NCOOR e a tranforma��o homog�nea afim T(3x3) 
    correspondente.
    
    NCOOR, T = normcoor(COOR, METHOD)
    Permite efectuar escalamentos do tipo :
      METHOD =
      'nnor' : (no normalization). Devolve as coordenadas de entrada,
               com T = eye(3)
      'none' : nenhum escalamento. 
      'stdv' : desvio padr�o unit�rio e independente em cada coordenada.
      'equa' : desvio padr�o aproximadamente unit�rio e igual para as duas
               coordenadas (m�dia dos desvios padroes).
      'sqrt' : desvio padr�o igual � raiz quadrada do d.p. original.
    
    COOR = trancoor(inv(T),NCOOR)
    """
    METHOD = METHOD.lower()
    assert METHOD in ("nnor", "stdv", "equa", "sqrt", "none"), (
        'METHOD must be one of "nnor", "stdv", "equa", "sqrt", "none"')
    if METHOD == "nnor":
        # Devolver as coordenadas de entrada 
        NCOOR = COOR
        T = np.eye(3)
        return NCOOR, T
    
    assert COOR.shape[1] == 2, "Coordinate array error"
    
    # Create the list of the repeated vector mean 
    MCOOR = np.mean(COOR, axis=0)
    
    if METHOD == "stdv":
        # Compute the vector standard deviation
        SDCOOR = np.mean((COOR - MCOOR)**2, axis=0)**0.5
    elif METHOD == 'equa':
        # Compute the average standard deviation
        SDCOOR = np.mean(
            np.mean((COOR - MCOOR)**2, axis=0), axis=0)*np.ones(2)
    elif METHOD == "sqrt":
        # Compute the sqr of the vector standard deviation
        SDCOOR = np.mean((COOR - MCOOR)**2, axis=0)
    elif METHOD == "none":
        SDCOOR = np.ones(2)
    
    # Account for data degeneracies, i. e., null st. dev.
    SDCOOR += np.any(SDCOOR < 1e-3) * 1e-3
    
    # Compute the homogeneous transformation
    transformation = np.eye(3)
    transformation[0, 0] = SDCOOR[0]**(-1)
    transformation[1, 1] = SDCOOR[1]**(-1)
    transformation[:2, 2] = -(MCOOR/SDCOOR)
    
    # Create homogeneous list
    COOR = np.hstack((COOR, np.ones((COOR.shape[0], 1))))
    
    # Get transformed list
    NCOOR = np.dot(COOR, transformation.T)
    
    # Divide by the homogeneous parameter
    NCOOR = NCOOR[:, :2]/NCOOR[:, np.newaxis, 2]
    
    return NCOOR, transformation


def lsplanar(COOR1, COOR2, normmethod='stdv'):
    """M12 = lsplanar(COOR1, COOR2)
    Determina��o da Transforma��o Planar M12(3x3) que relaciona os pontos
    com coordenadas COOR1 e COOR2.
    S�o necess�rias pelo menos 4 pontos em cada lista de coordenadas.
    
    M12, CONDS = lsplanar(COOR1, COOR2) devolve tamb�m o n� de condi��o 
    do sistema (A'*A).
    [M12,CONDS,A,R] = lsplanar(COOR1, COOR2) devolve a matriz do sistema 
    A e o vector de res�duos R.
    
    Usa o sistema (u,v) para as coordenadas 2-D.
    
    M12 = lsplanar(COOR1, COOR2, method)
    Permite a especifica��o de uma outra transforma��o planar:
      method = 'lsplanar' :  Transforma��o planar gen�rica.
    
    M12 = lsplanar(COOR1, COOR2, method, normmethod)
    Permite a especifica��o de um metodo para a normaliza�ao de coordenadas
    antes da estima�ao da transformacao planar:
      normmethod = 
          'none' : nenhum escalamento, so' transla�ao. 
          'stdv' : desvio padr�o unit�rio em cada coordenada.
          'sqrt' : desvio padr�o igual � raiz quadrada do d.p. original.
    """
    
    assert COOR1.shape == COOR2.shape, (
        "Coordinate lists must have the same size.")
    
    num_points = COOR1.shape[0]
    assert num_points > 1, (
        "Coordinate lists must have at least 1 matched points")
    
    # Teste com normaliza��o de coordenadas (parte 1/2)
    NCOOR1, T1 = normcoor(COOR1, normmethod)
    NCOOR2, T2 = normcoor(COOR2, normmethod)
    COOR1 = NCOOR1
    COOR2 = NCOOR2
    
    # No coordinate swapping
    ones_vec = np.ones((num_points, 1))
    COOR1 = np.hstack((COOR1, ones_vec))
    COOR2 = np.hstack((COOR2, ones_vec))
    
    #print "lsplanar():413"
    #code.interact(local=locals())
    
    A = np.zeros((2*num_points, 9))
    
    mat_1 = np.asarray([[1., 0., 0.]])
    mat_2 = np.asarray([[0., 1., 0.]])
    for l in range(num_points):
        mat_1[0, 2] = -COOR1[l, 0]
        mat_2[0, 2] = -COOR1[l, 1]
        ML1 = np.dot(COOR2[l, np.newaxis].T, mat_1)
        ML2 = np.dot(COOR2[l, np.newaxis].T, mat_2)
        A[2*l] = ML1.T.flatten()
        A[2*l+1] = ML2.T.flatten()
    
    #print "lsplanar():428"
    #code.interact(local=locals())
    
    # get the unit eigvector of A'*A corresponding to the smallest
    # eigenvalue of A'*A.
    N, CONDS = uniteig(A)
    if np.isinf(CONDS):
        print('A condi��o de A�*A � infinita !')
    
    # compute the residuals
    R = np.dot(A, N)
    #M12 = reshape(N(:,1),3,3)';
    M12 = N.reshape((3, 3))
    
    # Teste com normaliza��o de coordenadas (parte 2/2)
    M12 = np.dot(np.linalg.inv(T1), np.dot(M12, T2))
    
    #print "lsplanar():441"
    #code.interact(local=locals())
    return M12, CONDS, A, R


def uniteig(A):
    """N = uniteig(A)
    Get Unit Eigenvector of A'*A :
    Devolve o vector pr�prio unit�rio de A'*A correspondente ao menor
    valor pr�prio de A�*A. 
    
    N,CONDNUM = uniteig(A) :  Devolve tamb�m o n� de condi��o de A'*A. 
    """
    # get the unit eigvector of A'*A corresponding to the 
    # smallest eigenvalue of A.
    AT_A = np.dot(A.T, A)
    D, V = np.linalg.eigh(AT_A)
    I = D.argmin()
    N = V[:, I]
    N /= np.sqrt((N**2).sum())
    
    CONDNUM = np.linalg.cond(AT_A)
    
    if np.isinf(CONDNUM):
        print('UNITEIG : A condi��o de A�*A � infinita !')
    #print "uniteig():461"
    #code.interact(local=locals())
    return N, CONDNUM


def getposeparam_sturm(K, MiW, wtcflag="CTW"):
    """POSEPARAM = getposeparam(K,MiW)
    Pose Parameters from Homography using Sturm method:
    
    (falta help) 
    Paper Algorithms for Plane--Based Pose Estimation
    bib_nreg.bib - Sturm00
    
    (help antigo)
     
    Calcula os par�metros de pose de uma camera com matrix de parametros
    intr�nsecos K, e com uma homografia MiW entre a imagem e o plano de
    refer�ncia do mundo. Devolve os par�metros de pose
    POSEPAR = [a b c tx ty tz]. Os �ngulos a, b e c s�o os �ngulos 
    fixos X-Y-Z tal como definidos no Craig 2� Ed. Pag 46, para a rota��o 
    cRw. Os elementos tx, ty e tz correspondem � transla��o do ref da camera
    cTw. Para cada valor de MiW e K existem duas solu��es possiveis para a
    pose. Esta fun��o devolve a solu��o correspondente � componente positiva
    de wTc segundo ZZ, i.e., centro �ptico acima do plano do referencial do
    mundo. O referencial 3D do plano � coincidente com o 2D com o eixo
    adicional ZZ definido da maneira convencional a partir dos dois outros. 
    
    POSEPARAM = getposeparam(K,MiW,'WTC')
    Devolve os par�metros de pose na forma
    POSEPAR = [a b c wtcx wtcy wtcz], na qual os elementos wtcx, 
    wtcy e wtcz correspondem � transla��o do ref da camera wTc.
    
    Efectua a opera��o inversa de POSE1HOMO.M
    """ 

    if (K[1, 0]**2 + K[2, 0]**2 + K[2, 1]**2) > 1e9:
        print 'K should be a upper triangular matrix on GETPOSEPARAM.'
    wtcflag = wtcflag.upper()
    assert wtcflag in ["CTW", "WTC"], "wtcflag must be 'CTW' or 'WTC'"
    
    A = np.dot(np.linalg.inv(K), MiW)
    
    # Compute the "economy size" SVD
    U, S, V = np.linalg.svd(A[:, :2], full_matrices=0)
    
    L = np.dot(U, V.T)
    
    ln3 = np.cross(L[:, 0], L[:, 1])
    CRW1 = np.hstack((L, ln3[:, np.newaxis]))
    
    #Lambda = trace(S)/2;
    Lambda = S.sum()/2.
    
    CTWORG1 = A[:, 2] / Lambda
    WTCORG1 = np.dot(-CRW1.T, CTWORG1)
    
    #if WTCORG1[2] > 0:
    #    CRW2 = np.dot(CRW1, np.diag([-1., -1., 1.]))
    #    WTCORG2 = np.dot(np.diag([1., 1., -1.]), WTCORG1)
    #else:
    #    CRW2 = CRW1
    #    WTCORG2 = WTCORG1
    #    CRW1 = np.dot(CRW1, np.diag([-1., -1., 1.]))
    #    WTCORG1 = np.dot(np.diag([1., 1., -1.]), WTCORG1)
    #    CTWORG1 = -CTWORG1
    
    a, b, c = rot2xyzfixed(CRW1)
    
    if wtcflag == 'CTW':
        POSEPARAM = np.hstack((a, b, c, CTWORG1[:3]))
    else:
        POSEPARAM = np.hstack((a, b, c, WTCORG1[:3]))
    
    return POSEPARAM



def rot2xyzfixed(ROT3):
    """a, b, c = rot2xyzfixed(ROT3)
    Rotation to X-Y-Z fixed angles
    Calcula os �ngulos fixos de rota��o X-Y-Z a partir de uma matriz de 
    rota��o 3x3, de acordo com as f�rmulas do Craig 2� Ed. Pag. 47.
    """
    assert ROT3.shape == (3, 3), "Rotation matrix must be a 3x3 matrix"
    
    b = np.arctan2(-ROT3[2, 0], np.sqrt((ROT3[0, 0])**2 + (ROT3[1, 0])**2))
    cos_b = np.cos(b)
    # Check for degeneracy
    if abs(cos_b) < 1e-9:
        print "Degenerate angle set on angle extraction from rotation matrix."
        print "Cos(b) = ", cos_b
        b = np.sign(b) * np.pi/2
        a = 0.
        c = np.sign(b) * np.arctan2(ROT3[0, 1], ROT3[1, 1])
    else:
        a = np.arctan2(ROT3[1, 0]/cos_b, ROT3[0, 0]/cos_b)
        c = np.arctan2(ROT3[2, 1]/cos_b, ROT3[2, 2]/cos_b)
    return a, b, c


test_data = scipy.io.loadmat("/home/snagappa/udg/ros/udg_pandora/src/pose_cov/test_data.mat")
coori = test_data["coori"]
coorm = test_data["coorm"]
camera_matrix = test_data["K"]
wNm = test_data["MWR_norm"]
pixel_noise_var = test_data["sigmai"][0, 0]
homography_matrix = None
camera_cov = test_data["KCOV"]
wNm_cov = test_data["COVMWR"]

print "result = objdetect.compute_pose_unc(coori, coorm, camera_matrix, wNm, pixel_noise_var, homography_matrix, camera_cov, wNm_cov)"
result = compute_pose_unc(coori, coorm, camera_matrix, wNm, pixel_noise_var, homography_matrix, camera_cov, wNm_cov)
