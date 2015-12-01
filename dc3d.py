import numpy as np
import math
from math import sin, cos, tan, atan, radians
from math import sqrt, log

PI = math.pi
PI2 = PI*2

class C0_t:
    __slots__ = ['alp1', 'alp2', 'alp3', 'alp4', 'alp5', 'sd', 'cd', 'sdsd',
        'cdcd', 'sdcd', 's2d', 'c2d']

    def __init__(self, *args):
        for i, attr in enumerate(self.__slots__):
            setattr(self, attr, args[i])

class C1_t:
    __slots__ = ['p', 'q', 's', 't', 'xy', 'x2', 'y2', 'd2', 'r', 'r2', 'r3',
        'r5', 'qr', 'qrx', 'a3', 'a5', 'b3', 'c3', 'uy', 'vy', 'wy',
        'uz', 'vz', 'wz']

    def __init__(self, *args):
        for i, attr in enumerate(self.__slots__):
            setattr(self, attr, args[i])

class C2_t:
    __slots__ = ['xi2', 'et2', 'q2', 'r', 'r2', 'r3', 'r5', 'y', 'd', 'tt',
        'alx', 'ale', 'x11', 'y11', 'x32', 'y32', 'ey', 'ez', 'fy', 'fz',
        'gy', 'gz', 'hy', 'hz']

    def __init__(self, *args):
        for i, attr in enumerate(self.__slots__):
            setattr(self, attr, args[i])


def isclose(a, b, atol=1e-5, rtol=1e-8):
    if abs(a-b) <= (atol + rtol*abs(b)):
        return True
    else:
        return False


def nearlyzero(a):
    return isclose(a, 0.0)


def dc3d0(alpha,x,y,z,depth,dip,pot1,pot2,pot3,pot4):
# c                                                                       00060000
# c********************************************************************   00070000
# c*****                                                          *****   00080000
# c*****    displacement and strain at depth                      *****   00090000
# c*****    due to buried point source in a semiinfinite medium   *****   00100000
# c*****                         coded by  y.okada ... sep.1991   *****   00110002
# c*****                         revised   y.okada ... nov.1991   *****   00120002
# c*****                                                          *****   00130000
# c********************************************************************   00140000
# c                                                                       00150000
# c***** input                                                            00160000
# c*****   alpha : medium constant  (lambda+myu)/(lambda+2*myu)           00170000
# c*****   x,y,z : coordinate of observing point                          00180000
# c*****   depth : source depth                                           00190000
# c*****   dip   : dip-angle (degree)                                     00200000
# c*****   pot1-pot4 : strike-, dip-, tensile- and inflate-potency        00210000
# c*****       potency=(  moment of double-couple  )/myu     for pot1,2   00220000
# c*****       potency=(intensity of isotropic part)/lambda  for pot3     00230000
# c*****       potency=(intensity of linear dipole )/myu     for pot4     00240000
# c                                                                       00250000
# c***** output                                                           00260000
# c*****   ux, uy, uz  : displacement ( unit=(unit of potency) /          00270000
# c*****               :                     (unit of x,y,z,depth)**2  )  00280000
# c*****   uxx,uyx,uzx : x-derivative ( unit= unit of potency) /          00290000
# c*****   uxy,uyy,uzy : y-derivative        (unit of x,y,z,depth)**3  )  00300000
# c*****   uxz,uyz,uzz : z-derivative                                     00310000
# c*****   iret        : return code  ( =0....normal,   =1....singular )  00320002
# c                                                                       00330000
    u = np.zeros((12))
    dua = np.zeros((12))
    dub = np.zeros((12))
    duc = np.zeros((12))

    if z > 0.:
        print('0** positive z was given in sub-dc3d0')

    aalpha=alpha
    ddip=dip
    C0 = dccon0(aalpha,ddip)
    #c======================================                                 00480000
    #c=====  real-source contribution  =====                                 00490000
    #c======================================                                 00500000
    xx=x
    yy=y
    zz=z
    dd=depth+z
    C1 = dccon1(xx,yy,dd,C0)

    #c=======================================                                00960000
    #c=====  in case of singular (r=0)  =====                                00970000
    #c=======================================                                00980000
    r = C1.r
    if r == 0.:
        return 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.

    pp1=pot1
    pp2=pot2
    pp3=pot3
    pp4=pot4
    dua = ua0(xx,yy,dd,pp1,pp2,pp3,pp4, C0, C1)
    #c-----                                                                  00620000
    u[:9] -= dua[:9]
    u[9:] += dua[9:]
    #c=======================================                                00670000
    #c=====  image-source contribution  =====                                00680000
    #c=======================================                                00690000
    dd=depth-z
    C1 = dccon1(xx,yy,dd, C0)
    dua = ua0(xx,yy,dd,pp1,pp2,pp3,pp4, C0, C1)
    dub = ub0(xx,yy,dd,zz,pp1,pp2,pp3,pp4, C0, C1)
    duc = uc0(xx,yy,dd,zz,pp1,pp2,pp3,pp4, C0, C1)
    #c-----                                                                  00750000
    du = dua + dub + zz*duc
    du[9:] += duc[:3]
    u += du
    #c=====                                                                  00810000
    ux=u[0]
    uy=u[1]
    uz=u[2]
    uxx=u[3]
    uyx=u[4]
    uzx=u[5]
    uxy=u[6]
    uyy=u[7]
    uzy=u[8]
    uxz=u[9]
    uyz=u[10]
    uzz=u[11]
    iret=0
    return ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz, iret


def dc3d(alpha, x, y, z, depth, dip, al1, al2, aw1, aw2, disl1, disl2, disl3):
#C                                                                       04670005
#C********************************************************************   04680005
#C*****                                                          *****   04690005
#C*****    DISPLACEMENT AND STRAIN AT DEPTH                      *****   04700005
#C*****    DUE TO BURIED FINITE FAULT IN A SEMIINFINITE MEDIUM   *****   04710005
#C*****              CODED BY  Y.OKADA ... SEP.1991              *****   04720005
#C*****              REVISED ... NOV.1991, APR.1992, MAY.1993,   *****   04730005
#C*****                          JUL.1993                        *****   04740005
#C********************************************************************   04750005
#C                                                                       04760005
#C***** INPUT                                                            04770005
#C*****   ALPHA : MEDIUM CONSTANT  (LAMBDA+MYU)/(LAMBDA+2*MYU)           04780005
#C*****   X,Y,Z : COORDINATE OF OBSERVING POINT                          04790005
#C*****   DEPTH : DEPTH OF REFERENCE POINT                               04800005
#C*****   DIP   : DIP-ANGLE (DEGREE)                                     04810005
#C*****   AL1,AL2   : FAULT LENGTH RANGE                                 04820005
#C*****   AW1,AW2   : FAULT WIDTH RANGE                                  04830005
#C*****   DISL1-DISL3 : STRIKE-, DIP-, TENSILE-DISLOCATIONS              04840005
#C                                                                       04850005
#C***** OUTPUT                                                           04860005
#C*****   UX, UY, UZ  : DISPLACEMENT ( UNIT=(UNIT OF DISL)               04870005
#C*****   UXX,UYX,UZX : X-DERIVATIVE ( UNIT=(UNIT OF DISL) /             04880005
#C*****   UXY,UYY,UZY : Y-DERIVATIVE        (UNIT OF X,Y,Z,DEPTH,AL,AW) )04890005
#C*****   UXZ,UYZ,UZZ : Z-DERIVATIVE                                     04900005
#C*****   IRET        : RETURN CODE  ( =0....NORMAL,   =1....SINGULAR )  04910005
#C
    if z > 0.:
        print('( ** POSITIVE Z WAS GIVEN IN SUB-DC3D)')

    xi = np.zeros((2))
    et = np.zeros((2))
    kxi = np.zeros((2))
    ket = np.zeros((2))
    u = np.zeros((12))
    du = np.zeros((12))
    dua = np.zeros((12))
    dub = np.zeros((12))
    duc = np.zeros((12))

    aalpha = alpha
    ddip = dip

    C0 = dccon0(aalpha, ddip)

    sd, cd = C0.sd, C0.cd
    zz = z
    dd1 = disl1
    dd2 = disl2
    dd3 = disl3

    xi[0] = x - al1
    xi[1] = x - al2
    if nearlyzero(xi[0]):
        xi[0] = 0.0
    if nearlyzero(xi[1]):
        xi[1] = 0.0

    #======================================
    #=====  REAL-SOURCE CONTRIBUTION  =====
    #======================================
    d = depth + z
    p = y*cd + d*sd
    q = y*sd - d*cd
    et[0] = p - aw1
    et[1] = p - aw2
    if nearlyzero(et[0]):
        et[0] = 0.0
    if nearlyzero(et[1]):
        et[1] = 0.0

    #--------------------------------
    #----- REJECT SINGULAR CASE -----
    #--------------------------------
    #C----- ON FAULT EDGE
    if q == 0.0 and \
        ((xi[0]*xi[1] <= 0.0 and et[0]*et[1] == 0.0) or \
         (et[0]*et[1] <= 0.0 and xi[0]*xi[1] == 0.0)):
    #*   GO TO 99                                                       05350005
        return None # -> modified later, return 0s

    #C----- ON NEGATIVE EXTENSION OF FAULT EDGE                              05360014
    r12 = sqrt(xi[0]*xi[0] + et[1]*et[1] + q*q)
    r21 = sqrt(xi[1]*xi[1] + et[0]*et[0] + q*q)
    r22 = sqrt(xi[1]*xi[1] + et[1]*et[1] + q*q)

    if xi[0] < 0.0 and r21+xi[1] < 1.e-6:
        kxi[0] = 1
    if xi[0] < 0.0 and r22+xi[1] < 1.e-6:
        kxi[1] = 1
    if et[0] < 0.0 and r12+et[1] < 1.e-6:
        ket[0] = 1
    if et[0] < 0.0 and r22+et[1] < 1.e-6:
        ket[1] = 1

    for k in range(2):
        for j in range(2):
            C2 = dccon2(xi[j], et[k], q, sd, cd, kxi[k], ket[j])
            dua = ua(xi[j], et[k], q, dd1, dd2, dd3, C0, C2)
            for i in [0, 3, 6, 9]:
                du[i] = -dua[i]
                du[i+1] = -dua[i+1]*cd + dua[i+2]*sd
                du[i+2] = -dua[i+1]*sd - dua[i+2]*cd
                if i < 9:
                    continue
                du[i] = -du[i]
                du[i+1] = -du[i+1]
                du[i+2] = -du[i+2]

            for i in range(12):
                #if j+k /= 3:
                if j+k != 1:
                    u[i] += du[i]
                #if j+k == 3:
                if j+k == 1:
                    u[i] -= du[i]

    # C=======================================                                05700005
    # C=====  IMAGE-SOURCE CONTRIBUTION  =====                                05710005
    # C=======================================                                05720005
    d = depth - z
    p = y*cd + d*sd
    q = y*sd - d*cd
    et[0] = p - aw1
    et[1] = p - aw2
    if nearlyzero(q):
        q = 0.0
    if nearlyzero(et[0]):
        et[0] = 0.0
    if nearlyzero(et[1]):
        et[1] = 0.0

    # c--------------------------------                                       05810005
    # c----- reject singular case -----                                       05820005
    # c--------------------------------                                       05830005
    # c----- on fault edge                                                    05840015
    if q == 0.0 and \
        (    (xi[0]*xi[1] <= 0.0 and et[0]*et[1] == 0.0) \
          or (et[0]*et[1] <= 0.0 and xi[0]*xi[1] == 0.0) ):
#      *   go to 99                                                       05880015
        return None # -> modified later, return 0s

    ## c----- on negative extension of fault edge                              05890015
    kxi[:] = 0 # init flag
    ket[:] = 0
    r12 = sqrt(xi[0]*xi[0]+et[1]*et[1]+q*q)
    r21 = sqrt(xi[1]*xi[1]+et[0]*et[0]+q*q)
    r22 = sqrt(xi[1]*xi[1]+et[1]*et[1]+q*q)
    if xi[0] < 0.0 and r21+xi[1] < 1.e-6:
        kxi[0] = 1
    if xi[0] < 0.0 and r22+xi[1] < 1.e-6:
        kxi[1] = 1
    if et[0] < 0.0 and r12+et[1] < 1.e-6:
        ket[0] = 1
    if et[0] < 0.0 and r22+et[1] < 1.e-6:
        ket[1] = 1

    for k in range(2):
        for j in range(2):
            C2 = dccon2(xi[j],et[k],q,sd,cd,kxi[k],ket[j])
            dua = ua(xi[j],et[k],q,dd1,dd2,dd3,C0, C2)
            dub = ub(xi[j],et[k],q,dd1,dd2,dd3,C0, C2)
            duc = uc(xi[j],et[k],q,zz,dd1,dd2,dd3,C0, C2)

            for i in [0, 3, 6, 9]:
                du[i] = dua[i] + dub[i] +z*duc[i]
                du[i+1] = (dua[i+1]+dub[i+1]+z*duc[i+1])*cd \
                    -(dua[i+2]+dub[i+2]+z*duc[i+2])*sd
                du[i+2] = (dua[i+1]+dub[i+1]-z*duc[i+1])*sd \
                    +(dua[i+2]+dub[i+2]-z*duc[i+2])*cd
                if i < 9:
                    continue
                du[i] +=  duc[0]
                du[i+1] += duc[1]*cd-duc[2]*sd
                du[i+2] += -duc[1]*sd-duc[2]*cd

            for i in range(12):
                if j+k != 1:
                    u[i] += du[i]
                if j+k == 1:
                    u[i] -= du[i]

    ###
    ux = u[0]
    uy = u[1]
    uz = u[2]
    uxx = u[3]
    uyx = u[4]
    uzx = u[5]
    uxy = u[6]
    uyy = u[7]
    uzy = u[8]
    uxz = u[9]
    uyz = u[10]
    uzz = u[11]
    iret = 0

    return ux, uy, uz, uxx, uyx, uzx, uxy, uyy, uzy, uxz, uyz, uzz, iret


def dccon0(alpha, dip):
    '''
    Calculate medium constants and fault-dip constants

    INPUT:
        alpha : medium constant (lambda+mu)/(lambda+2*mu)
        dip   : dip angle (degree)

    RETURN:
        alp1, alp2, alp3, alp4, alp5,
        sd, cd, sdsd, cdcd, sdcd, s2d, c2d

    ### CAUTION ###
      if cos(dip) is sufficiently small, it is set to zero.
    '''
    alp1 = (1.0 - alpha)/2.0
    alp2 = alpha/2.0
    alp3 = (1.0 - alpha)/alpha
    alp4 = 1.0 - alpha
    alp5 = alpha

    sd = sin(radians(dip))
    cd = cos(radians(dip))

    if nearlyzero(cd):
        cd = 0.0
        if sd > 0.:
            sd = 1.0
        elif sd < 0.:
            sd = -1.0

    sdsd = sd*sd
    cdcd = cd*cd
    sdcd = sd*cd
    s2d = 2.0*sdcd
    c2d = cdcd - sdsd

    return C0_t(alp1, alp2, alp3, alp4, alp5, sd, cd, sdsd, cdcd, sdcd, s2d, c2d)


def dccon1(x, y, d, C0):
    '''calculate station geometry constants for point source.

    INPUT:
        x,y,d : station coordinates in fault system
    '''
    sd = C0.sd
    cd = C0.cd
    sdsd = C0.sdsd
    cdcd = C0.cdcd
    sdcd = C0.sdcd
    s2d = C0.s2d
    c2d = C0.c2d

    if nearlyzero(x):
        x = 0.0
    if nearlyzero(y):
        y = 0.0
    if nearlyzero(d):
        d = 0.0

    p=y*cd+d*sd
    q=y*sd-d*cd
    s=p*sd+q*cd
    t=p*cd-q*sd
    xy=x*y
    x2=x*x
    y2=y*y
    d2=d*d
    r2=x2+y2+d2
    r =sqrt(r2)

    if r == 0.:
        return

    r3=r *r2
    r5=r3*r2
    r7=r5*r2
    #c-----                                                                  10390005
    a3=1.-3.*x2/r2
    a5=1.-5.*x2/r2
    b3=1.-3.*y2/r2
    c3=1.-3.*d2/r2
    #c-----                                                                  10440005
    qr=3.*q/r5
    qrx=5.*qr*x/r2
    #c-----                                                                  10470005
    uy=sd-5.*y*q/r2
    uz=cd+5.*d*q/r2
    vy=s -5.*y*p*q/r2
    vz=t +5.*d*p*q/r2
    wy=uy+sd
    wz=uz+cd
    return C1_t(p, q, s, t, xy, x2, y2, d2, r, r2, r3, r5, qr, qrx, a3, a5, b3,
        c3, uy, vy, wy, uz, vz, wz)


def dccon2(xi, et, q, sd, cd, kxi, ket):
    '''
    Calculate station geometry constants for finite source

    INPUT:
        xi, et, q : station coordinates in fault system
        sd, cd    : sin, cos of dip-angel
        kxi, ket  : kxi=1, ket=1 means r+xi<eps, r+et<eps, respectively.

    ### CAUTION ###
      if xi, et, q are sufficiently small, they are set to zero.
    '''
    if nearlyzero(xi):
        xi = 0.0
    if nearlyzero(et):
        et = 0.0
    if nearlyzero(q):
        q = 0.0

    xi2 = xi*xi
    et2 = et*et
    q2 = q*q
    r2 = xi2 + et2 + q2
    r = sqrt(r2)

    if nearlyzero(r):
        return

    r3 = r*r2
    r5 = r3*r2
    y = et*cd + q*sd
    d = et*sd - q*cd

    if q == 0.0:
        tt = 0.0
    else:
        tt = atan(xi*et/(q*r))

    if kxi == 1:
        alx = -log(r-xi)
        x11 = 0.0
        x32 = 0.0
    else:
        rxi = r + xi
        alx = log(rxi)
        x11 = 1.0/(r*rxi)
        x32 = (r+rxi)*x11*x11/r

    if ket == 1:
        ale = -log(r-et)
        y11 = 0.0
        y32 = 0.0
    else:
        ret = r + et
        ale = log(ret)
        y11 = 1.0/(r*ret)
        y32 = (r+ret)*y11*y11/r

    ey = sd/r - y*q/r3
    ez = cd/r + d*q/r3
    fy = d/r3 + xi2*y32*sd
    fz = y/r3 + xi2*y32*cd
    gy = 2.0*x11*sd - y*q*x32
    gz = 2.0*x11*cd + d*q*x32
    hy = d*q*x32 + xi*q*y32*sd
    hz = y*q*x32 + xi*q*y32*cd

    return C2_t(xi2, et2, q2, r, r2, r3, r5, y, d, tt, alx, ale, x11, y11, x32,
        y32, ey, ez, fy, fz, gy, gz, hy, hz)


def ua0(x,y,d,pot1,pot2,pot3,pot4, C0, C1):
    ''' displacement and strain at depth (part-a)
    due to buried point source in a semiinfinite medium.

    INPUT:
        x,y,d : station coordinates in fault system
        pot1-pot4 : strike-, dip-, tensile- and inflate-potency
    OUTPUT:
        u(12) : displacement and their derivatives
    '''
    u = np.zeros((12))
    du = np.zeros((12))

    alp1 = C0.alp1
    alp2 = C0.alp2
    sd = C0.sd
    cd = C0.cd
    s2d = C0.s2d
    c2d = C0.c2d

    p = C1.p
    q = C1.q
    s = C1.s
    t = C1.t
    xy = C1.xy
    x2 = C1.x2
    r3 = C1.r3
    r5 = C1.r5
    qr = C1.qr
    qrx = C1.qrx
    a3 = C1.a3
    a5 = C1.a5
    b3 = C1.b3
    c3 = C1.c3
    uy = C1.uy
    vy = C1.vy
    wy = C1.wy
    uz = C1.uz
    vz = C1.vz
    wz = C1.wz

    #c======================================                                 01370000
    #c=====  strike-slip contribution  =====                                 01380000
    #c======================================                                 01390000
    if pot1 != 0.:
        du[ 0]= alp1*q/r3    +alp2*x2*qr
        du[ 1]= alp1*x/r3*sd +alp2*xy*qr
        du[ 2]=-alp1*x/r3*cd +alp2*x*d*qr
        du[ 3]= x*qr*(-alp1 +alp2*(1.+a5) )
        du[ 4]= alp1*a3/r3*sd +alp2*y*qr*a5
        du[ 5]=-alp1*a3/r3*cd +alp2*d*qr*a5
        du[ 6]= alp1*(sd/r3-y*qr) +alp2*3.*x2/r5*uy
        du[ 7]= 3.*x/r5*(-alp1*y*sd +alp2*(y*uy+q) )
        du[ 8]= 3.*x/r5*( alp1*y*cd +alp2*d*uy )
        du[ 9]= alp1*(cd/r3+d*qr) +alp2*3.*x2/r5*uz
        du[10]= 3.*x/r5*( alp1*d*sd +alp2*y*uz )
        du[11]= 3.*x/r5*(-alp1*d*cd +alp2*(d*uz-q) )

        u += pot1/PI2*du
    #c===================================                                    01560000
    #c=====  dip-slip contribution  =====                                    01570000
    #c===================================                                    01580000
    if pot2 != 0.:
        du[ 0]=            alp2*x*p*qr
        du[ 1]= alp1*s/r3 +alp2*y*p*qr
        du[ 2]=-alp1*t/r3 +alp2*d*p*qr
        du[ 3]=                 alp2*p*qr*a5
        du[ 4]=-alp1*3.*x*s/r5 -alp2*y*p*qrx
        du[ 5]= alp1*3.*x*t/r5 -alp2*d*p*qrx
        du[ 6]=                          alp2*3.*x/r5*vy
        du[ 7]= alp1*(s2d/r3-3.*y*s/r5) +alp2*(3.*y/r5*vy+p*qr)
        du[ 8]=-alp1*(c2d/r3-3.*y*t/r5) +alp2*3.*d/r5*vy
        du[ 9]=                          alp2*3.*x/r5*vz
        du[10]= alp1*(c2d/r3+3.*d*s/r5) +alp2*3.*y/r5*vz
        du[11]= alp1*(s2d/r3-3.*d*t/r5) +alp2*(3.*d/r5*vz-p*qr)

        u += pot2/PI2*du
    #c========================================                               01750000
    #c=====  tensile-fault contribution  =====                               01760000
    #c========================================                               01770000
    if pot3 != 0.:
        du[ 0]= alp1*x/r3 -alp2*x*q*qr
        du[ 1]= alp1*t/r3 -alp2*y*q*qr
        du[ 2]= alp1*s/r3 -alp2*d*q*qr
        du[ 3]= alp1*a3/r3     -alp2*q*qr*a5
        du[ 4]=-alp1*3.*x*t/r5 +alp2*y*q*qrx
        du[ 5]=-alp1*3.*x*s/r5 +alp2*d*q*qrx
        du[ 6]=-alp1*3.*xy/r5           -alp2*x*qr*wy
        du[ 7]= alp1*(c2d/r3-3.*y*t/r5) -alp2*(y*wy+q)*qr
        du[ 8]= alp1*(s2d/r3-3.*y*s/r5) -alp2*d*qr*wy
        du[ 9]= alp1*3.*x*d/r5          -alp2*x*qr*wz
        du[10]=-alp1*(s2d/r3-3.*d*t/r5) -alp2*y*qr*wz
        du[11]= alp1*(c2d/r3+3.*d*s/r5) -alp2*(d*wz-q)*qr

        u += pot3/PI2*du
    #c=========================================                              01940000
    #c=====  inflate source contribution  =====                              01950000
    #c=========================================                              01960000
    if pot4 != 0.:
        du[ 0]=-alp1*x/r3
        du[ 1]=-alp1*y/r3
        du[ 2]=-alp1*d/r3
        du[ 3]=-alp1*a3/r3
        du[ 4]= alp1*3.*xy/r5
        du[ 5]= alp1*3.*x*d/r5
        du[ 6]= du[4]
        du[ 7]=-alp1*b3/r3
        du[ 8]= alp1*3.*y*d/r5
        du[ 9]=-du[5]
        du[10]=-du[8]
        du[11]= alp1*c3/r3

        u += pot4/PI2*du
    return u


def ub0(x,y,d,z,pot1,pot2,pot3,pot4,C0, C1):
    '''displacement and strain at depth (part-b)
    due to buried point source in a semiinfinite medium

    INPUT:
        x,y,d,z : station coordinates in fault system
        pot1-pot4 : strike-, dip-, tensile- and inflate-potency
    OUTPUT:
        u(12) : displacement and their derivatives
    '''
    u = np.zeros((12))
    du = np.zeros((12))

    alp3 = C0.alp3
    sd = C0.sd
    sdsd = C0.sdsd
    sdcd = C0.sdcd

    p = C1.p
    q = C1.q
    xy = C1.xy
    x2 = C1.x2
    y2 = C1.y2
    d2 = C1.d2
    r = C1.r
    r2 = C1.r2
    r3 = C1.r3
    r5 = C1.r5
    qr = C1.qr
    qrx = C1.qrx
    a3 = C1.a3
    a5 = C1.a5
    b3 = C1.b3
    c3 = C1.c3
    uy = C1.uy
    vy = C1.vy
    wy = C1.wy
    uz = C1.uz
    vz = C1.vz
    wz = C1.wz

    c=d+z
    rd=r+d
    d12=1./(r*rd*rd)
    d32=d12*(2.*r+d)/r2
    d33=d12*(3.*r+d)/(r2*rd)
    d53=d12*(8.*r2+9.*r*d+3.*d2)/(r2*r2*rd)
    d54=d12*(5.*r2+4.*r*d+d2)/r3*d12

    fi1= y*(d12-x2*d33)
    fi2= x*(d12-y2*d33)
    fi3= x/r3-fi2
    fi4=-xy*d32
    fi5= 1./(r*rd)-x2*d32
    fj1=-3.*xy*(d33-x2*d54)
    fj2= 1./r3-3.*d12+3.*x2*y2*d54
    fj3= a3/r3-fj2
    fj4=-3.*xy/r5-fj1
    fk1=-y*(d32-x2*d53)
    fk2=-x*(d32-y2*d53)
    fk3=-3.*x*d/r5-fk2

    #c======================================                                 02600000
    #c=====  strike-slip contribution  =====                                 02610000
    #c======================================                                 02620000
    if pot1 != 0.:
        du[ 0]=-x2*qr  -alp3*fi1*sd
        du[ 1]=-xy*qr  -alp3*fi2*sd
        du[ 2]=-c*x*qr -alp3*fi4*sd
        du[ 3]=-x*qr*(1.+a5) -alp3*fj1*sd
        du[ 4]=-y*qr*a5      -alp3*fj2*sd
        du[ 5]=-c*qr*a5      -alp3*fk1*sd
        du[ 6]=-3.*x2/r5*uy      -alp3*fj2*sd
        du[ 7]=-3.*xy/r5*uy-x*qr -alp3*fj4*sd
        du[ 8]=-3.*c*x/r5*uy     -alp3*fk2*sd
        du[ 9]=-3.*x2/r5*uz  +alp3*fk1*sd
        du[10]=-3.*xy/r5*uz  +alp3*fk2*sd
        du[11]= 3.*x/r5*(-c*uz +alp3*y*sd)

        u += pot1/PI2*du
    #c===================================                                    02790000
    #c=====  dip-slip contribution  =====                                    02800000
    #c===================================                                    02810000
    if pot2 != 0.:
        du[ 0]=-x*p*qr +alp3*fi3*sdcd
        du[ 1]=-y*p*qr +alp3*fi1*sdcd
        du[ 2]=-c*p*qr +alp3*fi5*sdcd
        du[ 3]=-p*qr*a5 +alp3*fj3*sdcd
        du[ 4]= y*p*qrx +alp3*fj1*sdcd
        du[ 5]= c*p*qrx +alp3*fk3*sdcd
        du[ 6]=-3.*x/r5*vy      +alp3*fj1*sdcd
        du[ 7]=-3.*y/r5*vy-p*qr +alp3*fj2*sdcd
        du[ 8]=-3.*c/r5*vy      +alp3*fk1*sdcd
        du[ 9]=-3.*x/r5*vz -alp3*fk3*sdcd
        du[10]=-3.*y/r5*vz -alp3*fk1*sdcd
        du[11]=-3.*c/r5*vz +alp3*a3/r3*sdcd

        u += pot2/PI2*du
    #c========================================                               02980000
    #c=====  tensile-fault contribution  =====                               02990000
    #c========================================                               03000000
    if pot3 != 0.:
        du[ 0]= x*q*qr -alp3*fi3*sdsd
        du[ 1]= y*q*qr -alp3*fi1*sdsd
        du[ 2]= c*q*qr -alp3*fi5*sdsd
        du[ 3]= q*qr*a5 -alp3*fj3*sdsd
        du[ 4]=-y*q*qrx -alp3*fj1*sdsd
        du[ 5]=-c*q*qrx -alp3*fk3*sdsd
        du[ 6]= x*qr*wy     -alp3*fj1*sdsd
        du[ 7]= qr*(y*wy+q) -alp3*fj2*sdsd
        du[ 8]= c*qr*wy     -alp3*fk1*sdsd
        du[ 9]= x*qr*wz +alp3*fk3*sdsd
        du[10]= y*qr*wz +alp3*fk1*sdsd
        du[11]= c*qr*wz -alp3*a3/r3*sdsd

        u += pot3/PI2*du
    #c=========================================                              03170000
    #c=====  inflate source contribution  =====                              03180000
    #c=========================================                              03190000
    if pot4 != 0.:
        du[ 0]= alp3*x/r3
        du[ 1]= alp3*y/r3
        du[ 2]= alp3*d/r3
        du[ 3]= alp3*a3/r3
        du[ 4]=-alp3*3.*xy/r5
        du[ 5]=-alp3*3.*x*d/r5
        du[ 6]= du[4]
        du[ 7]= alp3*b3/r3
        du[ 8]=-alp3*3.*y*d/r5
        du[ 9]=-du[5]
        du[10]=-du[8]
        du[11]=-alp3*c3/r3

        u += pot4/PI2*du
    return u


def uc0(x,y,d,z,pot1,pot2,pot3,pot4, C0, C1):
    '''displacement and strain at depth (part-b)
    due to buried point source in a semiinfinite medium

    INPUT:
        x,y,d,z : station coordinates in fault system
        pot1-pot4 : strike-, dip-, tensile- and inflate-potency
    OUTPUT:
        u(12) : displacement and their derivatives
    '''
    u = np.zeros((12))
    du = np.zeros((12))

    alp4 = C0.alp4
    alp5 = C0.alp5
    sd = C0.sd
    cd = C0.cd
    sdsd = C0.sdsd
    sdcd = C0.sdcd
    s2d = C0.s2d
    c2d = C0.c2d

    p = C1.p
    s = C1.s
    t = C1.t
    xy = C1.xy
    x2 = C1.x2
    y2 = C1.y2
    d2 = C1.d2
    q = C1.q
    r2 = C1.r2
    r3 = C1.r3
    r5 = C1.r5
    qr = C1.qr
    qrx = C1.qrx
    a3 = C1.a3
    a5 = C1.a5
    c3 = C1.c3

    c=d+z
    q2=q*q
    r7=r5*r2
    a7=1.-7.*x2/r2
    b5=1.-5.*y2/r2
    b7=1.-7.*y2/r2
    c5=1.-5.*d2/r2
    c7=1.-7.*d2/r2
    d7=2.-7.*q2/r2
    qr5=5.*q/r2
    qr7=7.*q/r2
    dr5=5.*d/r2

    #c======================================                                 03740000
    #c=====  strike-slip contribution  =====                                 03750000
    #c======================================                                 03760000
    if pot1 != 0.:
        du[ 0]=-alp4*a3/r3*cd  +alp5*c*qr*a5
        du[ 1]= 3.*x/r5*( alp4*y*cd +alp5*c*(sd-y*qr5) )
        du[ 2]= 3.*x/r5*(-alp4*y*sd +alp5*c*(cd+d*qr5) )
        du[ 3]= alp4*3.*x/r5*(2.+a5)*cd   -alp5*c*qrx*(2.+a7)
        du[ 4]= 3./r5*( alp4*y*a5*cd +alp5*c*(a5*sd-y*qr5*a7) )
        du[ 5]= 3./r5*(-alp4*y*a5*sd +alp5*c*(a5*cd+d*qr5*a7) )
        du[ 6]= du[4]
        du[ 7]= 3.*x/r5*( alp4*b5*cd -alp5*5.*c/r2*(2.*y*sd+q*b7) )
        du[ 8]= 3.*x/r5*(-alp4*b5*sd +alp5*5.*c/r2*(d*b7*sd-y*c7*cd) )
        du[ 9]= 3./r5*   (-alp4*d*a5*cd +alp5*c*(a5*cd+d*qr5*a7) )
        du[10]= 15.*x/r7*( alp4*y*d*cd  +alp5*c*(d*b7*sd-y*c7*cd) )
        du[11]= 15.*x/r7*(-alp4*y*d*sd  +alp5*c*(2.*d*cd-q*c7) )

        u += pot1/PI2*du
    #c===================================                                    03930000
    #c=====  dip-slip contribution  =====                                    03940000
    #c===================================                                    03950000
    if pot2 != 0.:
        du[ 0]= alp4*3.*x*t/r5          -alp5*c*p*qrx
        du[ 1]=-alp4/r3*(c2d-3.*y*t/r2) +alp5*3.*c/r5*(s-y*p*qr5)
        du[ 2]=-alp4*a3/r3*sdcd         +alp5*3.*c/r5*(t+d*p*qr5)
        du[ 3]= alp4*3.*t/r5*a5              -alp5*5.*c*p*qr/r2*a7
        du[ 4]= 3.*x/r5*(alp4*(c2d-5.*y*t/r2)-alp5*5.*c/r2*(s-y*p*qr7))
        du[ 5]= 3.*x/r5*(alp4*(2.+a5)*sdcd   -alp5*5.*c/r2*(t+d*p*qr7))
        du[ 6]= du[4]
        du[ 7]= 3./r5*(alp4*(2.*y*c2d+t*b5) \
                +alp5*c*(s2d-10.*y*s/r2-p*qr5*b7))
        du[ 8]= 3./r5*(alp4*y*a5*sdcd-alp5*c*((3.+a5)*c2d+y*p*dr5*qr7))
        du[ 9]= 3.*x/r5*(-alp4*(s2d-t*dr5) -alp5*5.*c/r2*(t+d*p*qr7))
        du[10]= 3./r5*(-alp4*(d*b5*c2d+y*c5*s2d) \
                -alp5*c*((3.+a5)*c2d+y*p*dr5*qr7))
        du[11]= 3./r5*(-alp4*d*a5*sdcd-alp5*c*(s2d-10.*d*t/r2+p*qr5*c7))

        u += pot2/PI2*du
    #c========================================                               04140000
    #c=====  tensile-fault contribution  =====                               04150000
    #c========================================                               04160000
    if pot3 != 0.:
        du[ 0]= 3.*x/r5*(-alp4*s +alp5*(c*q*qr5-z))
        du[ 1]= alp4/r3*(s2d-3.*y*s/r2)+alp5*3./r5*(c*(t-y+y*q*qr5)-y*z)
        du[ 2]=-alp4/r3*(1.-a3*sdsd)   -alp5*3./r5*(c*(s-d+d*q*qr5)-d*z)
        du[ 3]=-alp4*3.*s/r5*a5 +alp5*(c*qr*qr5*a7-3.*z/r5*a5)
        du[ 4]= 3.*x/r5*(-alp4*(s2d-5.*y*s/r2) \
                -alp5*5./r2*(c*(t-y+y*q*qr7)-y*z))
        du[ 5]= 3.*x/r5*( alp4*(1.-(2.+a5)*sdsd) \
                +alp5*5./r2*(c*(s-d+d*q*qr7)-d*z))
        du[ 6]= du[4]
        du[ 7]= 3./r5*(-alp4*(2.*y*s2d+s*b5) \
                -alp5*(c*(2.*sdsd+10.*y*(t-y)/r2-q*qr5*b7)+z*b5))
        du[ 8]= 3./r5*( alp4*y*(1.-a5*sdsd) \
                +alp5*(c*(3.+a5)*s2d-y*dr5*(c*d7+z)))
        du[ 9]= 3.*x/r5*(-alp4*(c2d+s*dr5) \
                +alp5*(5.*c/r2*(s-d+d*q*qr7)-1.-z*dr5))
        du[10]= 3./r5*( alp4*(d*b5*s2d-y*c5*c2d) \
                +alp5*(c*((3.+a5)*s2d-y*dr5*d7)-y*(1.+z*dr5)))
        du[11]= 3./r5*(-alp4*d*(1.-a5*sdsd) \
                -alp5*(c*(c2d+10.*d*(s-d)/r2-q*qr5*c7)+z*(1.+c5)))

        u += pot3/PI2*du
    #c=========================================                              04400000
    #c=====  inflate source contribution  =====                              04410000
    #c=========================================                              04420000
    if pot4 != 0.:
        du[ 0]= alp4*3.*x*d/r5
        du[ 1]= alp4*3.*y*d/r5
        du[ 2]= alp4*c3/r3
        du[ 3]= alp4*3.*d/r5*a5
        du[ 4]=-alp4*15.*xy*d/r7
        du[ 5]=-alp4*3.*x/r5*c5
        du[ 6]= du[4]
        du[ 7]= alp4*3.*d/r5*b5
        du[ 8]=-alp4*3.*y/r5*c5
        du[ 9]= du[5]
        du[10]= du[8]
        du[11]= alp4*3.*d/r5*(2.+c5)

        u += pot4/PI2*du
    return u


def ua(xi, et, q, disl1, disl2, disl3, C0, C2):
    '''
    Displacement and strain at depth (part-a)
    due to buries finite fault in a semiinfinite medium.

    INPUT:
        xi, et, q   : station coordintes in fault system
        disl1-disl3 : strike, dip, tensile dislocations
    OUTPUT:
        u(12) : displacement and their derivatives
    '''

    u = np.zeros((12))
    du = np.zeros((12))

    alp1, alp2, sd, cd = C0.alp1, C0.alp2, C0.sd, C0.cd
    y11 = C2.y11
    x11 = C2.x11
    xi2 = C2.xi2
    q2  = C2.q2
    r   = C2.r
    r3  = C2.r3
    ale = C2.ale
    y32 = C2.y32
    fy  = C2.fy
    y   = C2.y
    d   = C2.d
    tt  = C2.tt
    alx = C2.alx
    ey  = C2.ey
    ez  = C2.ez
    fz  = C2.fz
    gy  = C2.gy
    gz  = C2.gz
    hy  = C2.hy
    hz  = C2.hz

    xy = xi*y11
    qx = q*x11
    qy = q*y11

    #C======================================                                 06850005
    #C=====  STRIKE-SLIP CONTRIBUTION  =====                                 06860005
    #C======================================                                 06870005
    if disl1 != 0.0:
        du[0]=    tt/2.0 +alp2*xi*qy
        du[1]=           alp2*q/r
        du[2]= alp1*ale -alp2*q*qy
        du[3]=-alp1*qy  -alp2*xi2*q*y32
        du[4]=          -alp2*xi*q/r3
        du[5]= alp1*xy  +alp2*xi*q2*y32
        du[6]= alp1*xy*sd        +alp2*xi*fy+d/2.0*x11
        du[7]=                    alp2*ey
        du[8]= alp1*(cd/r+qy*sd) -alp2*q*fy
        du[9]= alp1*xy*cd        +alp2*xi*fz+y/2.0*x11
        du[10]=                    alp2*ez
        du[11]=-alp1*(sd/r-qy*cd) -alp2*q*fz

        u[:] += disl1/PI2*du[:]
    #C======================================                                 07040005
    #C=====    DIP-SLIP CONTRIBUTION   =====                                 07050005
    #C======================================                                 07060005
    if disl2 != 0.0:
        du[0]=           alp2*q/r
        du[1]=    tt/2.0 +alp2*et*qx
        du[2]= alp1*alx -alp2*q*qx
        du[3]=        -alp2*xi*q/r3
        du[4]= -qy/2.0 -alp2*et*q/r3
        du[5]= alp1/r +alp2*q2/r3
        du[6]=                      alp2*ey
        du[7]= alp1*d*x11+xy/2.0*sd +alp2*et*gy
        du[8]= alp1*y*x11          -alp2*q*gy
        du[9]=                      alp2*ez
        du[10]= alp1*y*x11+xy/2.0*cd +alp2*et*gz
        du[11]=-alp1*d*x11          -alp2*q*gz

        u[:] += disl2/PI2*du[:]
    #C========================================                               07230005
    #C=====  TENSILE-FAULT CONTRIBUTION  =====                               07240005
    #C========================================                               07250005
    if disl3 != 0.0:
        du[0]=-alp1*ale -alp2*q*qy
        du[1]=-alp1*alx -alp2*q*qx
        du[2]=    tt/2.0 -alp2*(et*qx+xi*qy)
        du[3]=-alp1*xy  +alp2*xi*q2*y32
        du[4]=-alp1/r   +alp2*q2/r3
        du[5]=-alp1*qy  -alp2*q*q2*y32
        du[6]=-alp1*(cd/r+qy*sd)  -alp2*q*fy
        du[7]=-alp1*y*x11         -alp2*q*gy
        du[8]= alp1*(d*x11+xy*sd) +alp2*q*hy
        du[9]= alp1*(sd/r-qy*cd)  -alp2*q*fz
        du[10]= alp1*d*x11         -alp2*q*gz
        du[11]= alp1*(y*x11+xy*cd) +alp2*q*hz

        u[:] += disl3/PI2*du[:]

    return u


def ub(xi, et, q, disl1, disl2, disl3, C0, C2):
    '''
    Displacement and strain at depth (part-b)
    due to buried finite fault in a semiinfinite medium.

    INPUT:
        xi, et, q : station coordinates in fault system
        disl1-disl3 : strike-, dip-, tensile-dislocations
    output:
        u(12) : displacement and their derivatives
    '''
    u = np.zeros((12))
    du = np.zeros((12))

    alp3 = C0.alp3
    sd   = C0.sd
    cd   = C0.cd
    sdsd = C0.sdsd
    cdcd = C0.cdcd
    sdcd = C0.sdcd
    xi2  = C2.xi2
    q2   = C2.q2
    r    = C2.r
    r3   = C2.r3
    y    = C2.y
    d    = C2.d
    tt   = C2.tt
    ale  = C2.ale
    x11  = C2.x11
    y11  = C2.y11
    y32  = C2.y32
    ey   = C2.ey
    ez   = C2.ez
    fy   = C2.fy
    fz   = C2.fz
    gy   = C2.gy
    gz   = C2.gz
    hy   = C2.hy
    hz   = C2.hz

    rd = r + d
    d11 = 1.0/(r*rd)
    aj2 = xi*y/rd*d11
    aj5 = -(d+y*y/rd)*d11
    if cd != 0.0:
        if xi == 0.0:
            ai4 = 0.0
        else:
          x = sqrt(xi2+q2)
          ai4 = 1.0/cdcd*( xi/rd*sdcd \
              + 2.0*atan((et*(x+q*cd)+x*(r+x)*sd)/(xi*(r+x)*cd)) )

        ai3=(y*cd/rd-ale+sd*log(rd))/cdcd
        ak1=xi*(d11-y11*sd)/cd
        ak3=(q*y11-y*d11)/cd
        aj3=(ak1-aj2*sd)/cd
        aj6=(ak3-aj5*sd)/cd
    else:
        rd2=rd*rd
        ai3=(et/rd+y*q/rd2-ale)/2.0
        ai4=xi*y/rd2/2.0
        ak1=xi*q/rd*d11
        ak3=sd/rd*(xi2*d11-1.0)
        aj3=-xi/rd2*(q2*d11-1.0/2.0)
        aj6=-y/rd2*(xi2*d11-1.0/2.0)

    xy=xi*y11
    ai1=-xi/rd*cd-ai4*sd
    ai2= log(rd)+ai3*sd
    ak2= 1.0/r+ak3*sd
    ak4= xy*cd-ak1*sd
    aj1= aj5*cd-aj6*sd
    aj4=-xy-aj2*cd+aj3*sd

    qx=q*x11
    qy=q*y11

    #c======================================                                 08030005
    #c=====  strike-slip contribution  =====                                 08040005
    #c======================================                                 08050005
    if disl1 != 0.0:
        du[0]=-xi*qy-tt -alp3*ai1*sd
        du[1]=-q/r      +alp3*y/rd*sd
        du[2]= q*qy     -alp3*ai2*sd
        du[3]= xi2*q*y32 -alp3*aj1*sd
        du[4]= xi*q/r3   -alp3*aj2*sd
        du[5]=-xi*q2*y32 -alp3*aj3*sd
        du[6]=-xi*fy-d*x11 +alp3*(xy+aj4)*sd
        du[7]=-ey          +alp3*(1.0/r+aj5)*sd
        du[8]= q*fy        -alp3*(qy-aj6)*sd
        du[9]=-xi*fz-y*x11 +alp3*ak1*sd
        du[10]=-ez          +alp3*y*d11*sd
        du[11]= q*fz        +alp3*ak2*sd

        u[:] += disl1/PI2*du[:]

    #c======================================                                 08220005
    #c=====    dip-slip contribution   =====                                 08230005
    #c======================================                                 08240005
    if disl2 != 0.0:
        du[0]=-q/r      +alp3*ai3*sdcd
        du[1]=-et*qx-tt -alp3*xi/rd*sdcd
        du[2]= q*qx     +alp3*ai4*sdcd
        du[3]= xi*q/r3     +alp3*aj4*sdcd
        du[4]= et*q/r3+qy  +alp3*aj5*sdcd
        du[5]=-q2/r3       +alp3*aj6*sdcd
        du[6]=-ey          +alp3*aj1*sdcd
        du[7]=-et*gy-xy*sd +alp3*aj2*sdcd
        du[8]= q*gy        +alp3*aj3*sdcd
        du[9]=-ez          -alp3*ak3*sdcd
        du[10]=-et*gz-xy*cd -alp3*xi*d11*sdcd
        du[11]= q*gz        -alp3*ak4*sdcd

        u[:] += disl2/PI2*du[:]

    #c========================================                               08410005
    #c=====  tensile-fault contribution  =====                               08420005
    #c========================================                               08430005
    if disl3 != 0.0:
        du[0]= q*qy           -alp3*ai3*sdsd
        du[1]= q*qx           +alp3*xi/rd*sdsd
        du[2]= et*qx+xi*qy-tt -alp3*ai4*sdsd
        du[3]=-xi*q2*y32 -alp3*aj4*sdsd
        du[4]=-q2/r3     -alp3*aj5*sdsd
        du[5]= q*q2*y32  -alp3*aj6*sdsd
        du[6]= q*fy -alp3*aj1*sdsd
        du[7]= q*gy -alp3*aj2*sdsd
        du[8]=-q*hy -alp3*aj3*sdsd
        du[9]= q*fz +alp3*ak3*sdsd
        du[10]= q*gz +alp3*xi*d11*sdsd
        du[11]=-q*hz +alp3*ak4*sdsd

        u[:] += disl3/PI2*du[:]

    return u


def uc(xi, et, q, z, disl1, disl2, disl3, C0, C2):
    '''
    displacement and strain at depth (part-c)
    due to buried finite fault in a semiinfinite medium

    INPUT:
        xi, et, q, z : station coordinates in fault system
        disl1-disl3  : strike-, dip-, tensile-dislocations
    OUTPUT:
        u(12) : displacement and their derivatives
    '''
    u = np.zeros((12))
    du = np.zeros((12))

    alp4 = C0.alp4
    alp5 = C0.alp5
    sd = C0.sd
    cd = C0.cd
    sdsd = C0.sdsd
    cdcd = C0.cdcd
    sdcd = C0.sdcd

    xi2 = C2.xi2
    et2 = C2.et2
    q2 = C2.q2
    r = C2.r
    r2 = C2.r2
    r3 = C2.r3
    r5 = C2.r5
    y = C2.y
    d = C2.d
    x11 = C2.x11
    y11 = C2.y11
    x32 = C2.x32
    y32 = C2.y32

    c = d + z
    x53 = (8.0*r2+9.0*r*xi+3.0*xi2)*x11*x11*x11/r2
    y53 = (8.0*r2+9.0*r*et+3.0*et2)*y11*y11*y11/r2
    h=q*cd-z
    z32=sd/r3-h*y32
    z53=3.0*sd/r5-h*y53
    y0=y11-xi2*y32
    z0=z32-xi2*z53
    ppy=cd/r3+q*y32*sd
    ppz=sd/r3-q*y32*cd
    qq=z*y32+z32+z0
    qqy=3.0*c*d/r5-qq*sd
    qqz=3.0*c*y/r5-qq*cd+q*y32
    xy=xi*y11
    qx=q*x11
    qy=q*y11
    qr=3.0*q/r5
    cqx=c*q*x53
    cdr=(c+d)/r3
    yy0=y/r3-y0*cd

    #c======================================                                 09050005
    #c=====  strike-slip contribution  =====                                 09060005
    #c======================================                                 09070005
    if disl1 != 0.0:
        du[ 0]= alp4*xy*cd           -alp5*xi*q*z32
        du[ 1]= alp4*(cd/r+2.0*qy*sd)-alp5*c*q/r3
        du[ 2]= alp4*qy*cd           -alp5*(c*et/r3-z*y11+xi2*z32)
        du[ 3]= alp4*y0*cd                  -alp5*q*z0
        du[ 4]=-alp4*xi*(cd/r3+2.0*q*y32*sd) +alp5*c*xi*qr
        du[ 5]=-alp4*xi*q*y32*cd            +alp5*xi*(3.0*c*et/r5-qq)
        du[ 6]=-alp4*xi*ppy*cd    -alp5*xi*qqy
        du[ 7]= alp4*2.0*(d/r3-y0*sd)*sd-y/r3*cd \
                                  -alp5*(cdr*sd-et/r3-c*y*qr)
        du[ 8]=-alp4*q/r3+yy0*sd  +alp5*(cdr*cd+c*d*qr-(y0*cd+q*z0)*sd)
        du[ 9]= alp4*xi*ppz*cd    -alp5*xi*qqz
        du[10]= alp4*2.0*(y/r3-y0*cd)*sd+d/r3*cd -alp5*(cdr*cd+c*d*qr)
        du[11]=         yy0*cd    -alp5*(cdr*sd-c*y*qr-y0*sdsd+q*z0*cd)

        u += disl1/PI2*du
    #c======================================                                 09250005
    #c=====    dip-slip contribution   =====                                 09260005
    #c======================================                                 09270005
    if disl2 != 0.0:
        du[ 0]= alp4*cd/r -qy*sd -alp5*c*q/r3
        du[ 1]= alp4*y*x11       -alp5*c*et*q*x32
        du[ 2]=     -d*x11-xy*sd -alp5*c*(x11-q2*x32)
        du[ 3]=-alp4*xi/r3*cd +alp5*c*xi*qr +xi*q*y32*sd
        du[ 4]=-alp4*y/r3     +alp5*c*et*qr
        du[ 5]=    d/r3-y0*sd +alp5*c/r3*(1.0-3.0*q2/r2)
        du[ 6]=-alp4*et/r3+y0*sdsd -alp5*(cdr*sd-c*y*qr)
        du[ 7]= alp4*(x11-y*y*x32) -alp5*c*((d+2.0*q*cd)*x32-y*et*q*x53)
        du[ 8]=  xi*ppy*sd+y*d*x32 +alp5*c*((y+2.0*q*sd)*x32-y*q2*x53)
        du[ 9]=      -q/r3+y0*sdcd -alp5*(cdr*cd+c*d*qr)
        du[10]= alp4*y*d*x32       -alp5*c*((y-2.0*q*sd)*x32+d*et*q*x53)
        du[11]=-xi*ppz*sd+x11-d*d*x32-alp5*c*((d-2.0*q*cd)*x32-d*q2*x53)

        u += disl2/PI2*du
    #c========================================                               09440005
    #c=====  tensile-fault contribution  =====                               09450005
    #c========================================                               09460005
    if disl3 != 0.0:
        du[ 0]=-alp4*(sd/r+qy*cd)   -alp5*(z*y11-q2*z32)
        du[ 1]= alp4*2.0*xy*sd+d*x11 -alp5*c*(x11-q2*x32)
        du[ 2]= alp4*(y*x11+xy*cd)  +alp5*q*(c*et*x32+xi*z32)
        du[ 3]= alp4*xi/r3*sd+xi*q*y32*cd+alp5*xi*(3.0*c*et/r5-2.0*z32-z0)
        du[ 4]= alp4*2.0*y0*sd-d/r3 +alp5*c/r3*(1.0-3.0*q2/r2)
        du[ 5]=-alp4*yy0           -alp5*(c*et*qr-q*z0)
        du[ 6]= alp4*(q/r3+y0*sdcd)   +alp5*(z/r3*cd+c*d*qr-q*z0*sd)
        du[ 7]=-alp4*2.0*xi*ppy*sd-y*d*x32 \
                          +alp5*c*((y+2.0*q*sd)*x32-y*q2*x53)
        du[ 8]=-alp4*(xi*ppy*cd-x11+y*y*x32) \
                          +alp5*(c*((d+2.0*q*cd)*x32-y*et*q*x53)+xi*qqy)
        du[ 9]=  -et/r3+y0*cdcd -alp5*(z/r3*sd-c*y*qr-y0*sdsd+q*z0*cd)
        du[10]= alp4*2.0*xi*ppz*sd-x11+d*d*x32 \
                          -alp5*c*((d-2.0*q*cd)*x32-d*q2*x53)
        du[11]= alp4*(xi*ppz*cd+y*d*x32) \
                          +alp5*(c*((y-2.0*q*sd)*x32+d*et*q*x53)+xi*qqz)

        u += disl3/PI2*du
    return u

if __name__ == "__main__":
    pass
