import cv2
import numpy as np
import matplotlib.pyplot as plt


def funConv(start, length, band, high):
    res = np.zeros(l, np.uint8)
    for i in range(start, start + band):
        if i >= length:
            break
        res[i] = high
    return res


# conv = np.convolve(sumg,[100,100,100,100,100,100,100,100])

def distances_hb(val, f):
    res = []
    temp = 0
    for i in f:
        if (i > val and temp != 1):
            res.append(1)
            temp = 1
        elif (i <= val and temp != 2):
            res.append(1)
            temp = 2
        else:
            res[len(res) - 1] += 1
    return res


def quadra(m):  # moyenne quadratique
    a = m
    np.power(a, 2)
    summ = sum(a)
    summ = summ * 1 / len(a)
    return np.sqrt(summ)




# classification barbar

def classific(dist): #envoi les centres de gravite des classes, et leur nombre d'elements [centre, nbr elem]
    sdist = dist
    cl = [[]]  # ensemble de classes
    echantillonage = 8 # defini la severiter dans le choix de classification ( + c'est petit, + on genere de class, + on est grand, + on est sensible au bruit)
    for i in sdist:
        while np.size(cl, 0) <= int(round(i / echantillonage, 0)):
            cl.append([])
        c = cl[int(round(i / echantillonage, 0))]
        c.append(i)
        cl[int(round(i / echantillonage, 0))] = c

         # definition des classes principales, [[nmbre d'elments],[multiple de 10 inferieur],[multiple de 10 superieur]]
    nbcl = []
    for i in cl:
        nbcl.append(len(i))

    coord = [[0, 0, 0]]
    pos = 0
    tmp = []
    for i in range(0, len(nbcl)):
        if nbcl[i] != 0:
            tmp = coord[pos]
            if tmp[0] == 0:
                tmp[1] = i
            tmp[0] += nbcl[i]
            tmp[2] = i
            coord[pos] = tmp
        elif coord[pos][0] != 0:
            coord.append([0, 0, 0])
            pos+=1

    print 'taille des classes [[nmbre d\'elments],[multiple de 10 inferieur],[multiple de 10 superieur]]'
    print coord

    g = [] #calcul des centres de gravites
    for clas in coord:
        g.append([0, clas[0]])
        for i in range(clas[1], clas[2]+1):
            pos = len(g)-1
            g[pos][0] += sum(cl[i])
        g[pos][0] /= clas[0]

    return g

def frontiere(g): #definie les frontiere entre les centres de gravite. g = [[centre graviter],[valeur sans importance]]
    fr = []
    for i in range(0,len(g)-1):
        fr.append((g[i+1][0]+g[i][0])/2)
    return fr


########################################################################################################################
########################################################################################################################
########################################################################################################################


# img = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output/opening_morph.png')

# gray = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output_pattern/edged.png', cv2.IMREAD_GRAYSCALE)
# gray = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output_pattern/dilated.png', cv2.IMREAD_GRAYSCALE)
gray = cv2.imread('./cell_counter/tests/contours/output_pattern/dilated.png', cv2.IMREAD_GRAYSCALE)


minLineLength = 200
maxLineGap = 15
lines = cv2.HoughLinesP(gray, 0.05, np.pi / 180, 100, 0, 100)  # ,minLineLength,maxLineGap)

# print lines
imgs = gray * 0

for (x1, x2, y1, y2) in lines[0]:
    cv2.line(imgs, (x1, x2), (y1, y2), (255, 255, 255), 1)

# cv2.imshow('haha',imgs)
# imgs = gray
print 'Hough termine'

h, l = imgs.shape

# # # # # # # # # # # # # # # #

# sumg = np.zeros(h, np.uint8)
# for y in range(0, h):
#     for x in range(0, l):
#         if imgs[y, x] > 126:
#             sumg[y] += 1
#
# print 'sum gauche termine'

# # # # # # # # # # # # # # # #

sumh = np.zeros(l, np.uint8)
for x in range(0, l):
    for y in range(0, h):
        if imgs[y, x] > 126:
            sumh[x] += 1

print 'sum haut termine'

# # # # # # # # # # # # # # # #


sumt = sumh

seuil = np.max(sumt) * 1/2
print 'seuil'
print seuil


dist = distances_hb(seuil, sumt)
cl = classific(dist)
fr = frontiere(cl)


print dist
print cl
print fr

print 'largeur moyen du carre :'
print 

plotseuil = []
for i in sumt:
    plotseuil.append(seuil)

plt.plot(sumt)
plt.plot(plotseuil)
# plt.plot(conv)
plt.ylabel('some numbers')
plt.show()
cv2.waitKey(0)
np.transpose(sumt)
sumt.tofile('foo.csv', sep='\n', format='%10.1f')
cv2.imwrite('B:/Projets/Python/houghlines5.jpg', imgs)
cv2.waitKey(0)

# faiblesse : le seuil (variable seuil) => definir une heuristigue
#             eventuellement, il y aurait aussi la variable echantillonage
