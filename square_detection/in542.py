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

def distances_hb(seuil, f):
    res = []
    temp = 0
    for i in f:
        if (i > seuil and temp != 1):
            res.append(1)
            temp = 1
        elif (i <= seuil and temp != 2):
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

def classific(dist): #envoi les centres de gravite des classes, et leur nombre d'elements [centre, nbr element assoier a ce centre]
    sdist = dist
    cl = [[]]  # ensemble de classes
    echantillonage = 3 # defini la severiter dans le choix de classification ( + c'est petit, + on genere de class, + on est grand, + on est sensible au bruit)

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

    # print 'taille des classes [[nmbre d\'elments],[multiple de 10 inferieur],[multiple de 10 superieur]]'
    # print coord

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


def whoIsSquare(classifics):
    toClass = list(classifics)
    c = [0, [0, 0]] # [val,[frontieres de classifictations]]
    sort = [[0,0,0], [0,0,0], [0,0,0]] # [val, nbrElements, position dans la classification]
    ireset = 0
    for s in sort:
        i = ireset
        for [val, nbrEl] in toClass:
            if s[1] <= nbrEl and nbrEl > 4:
                s[1] = nbrEl
                s[0] = val
                s[2] = i
            i += 1
        if len(toClass) > 1:
            toClass.pop(s[2]-ireset)
            ireset += 1


    f = frontiere(classifics)

    for [val, nbrEl, i] in sort:
        if c[0] <= val:
            c[0] = val
            if i == 0:
                c[1][0] = 0
                c[1][1] = f[i]
            elif i == len(f):
                c[1][0] = f[i-1]
                c[1][1] = f[i-1]*1000
            else:
                c[1][0] = f[i-1]
                c[1][1] = f[i]

    return c

def getPosition(sumHorG, seuil, whoSquare): # renvoi les positions de debut et de fin des carres de l'image. [[x1,x2],[x1,x2],...]
                        # ATTENTION !! le seuil DOIT etre le meme que celui utilise pour distances_hb(seuil, sumHorG) !!!!!
    axe = sumHorG
    target = whoSquare[1]
    square = []

    start = 0
    length = 1
    temp = 0

    square = []

    for i in range(0, len(axe)):
        if (axe[i] > seuil and temp != 1):
            if target[0] < length < target[1]:
                square.append([start, i]) # peut-etre i-1 ??
            start = i
            length = 1
            temp = 1
        elif (axe[i] <= seuil and temp != 2):
            if target[0] < length < target[1]:
                square.append([start, i]) # peut-etre i-1 ??
            start = i
            length = 1
            temp = 2
        else:
            length += 1

    return square

def getSquareShape(getpositionG, getpositionH): #renvoi la position des carres, [ [[point sup gauche],[point inf droit]], ...]
    shape = []

    for (x1, x2) in getpositionH:
        for(y1,y2) in getpositionG:
            shape.append([[x1, y1],[x2, y2]])

    return shape

########################################################################################################################
########################################################################################################################
########################################################################################################################


# img = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output/opening_morph.png')

# gray = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output/dilated.png', cv2.IMREAD_GRAYSCALE)
# origine = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output/opening_morph.png')


# gray = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output_pattern/edged.png', cv2.IMREAD_GRAYSCALE)
gray = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output_pattern/dilated.png', cv2.IMREAD_GRAYSCALE)
origine = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/contours_sample_3_raw.jpg')
# gray = cv2.imread('../tests/contours/output_pattern/dilated.png', cv2.IMREAD_GRAYSCALE)

kernel = np.ones((5,5),np.uint8)

cv2.imshow('jkhufsdeuhkivfdqs', gray)
cv2.waitKey(0)

for i in range(0,10):
    gray = cv2.dilate(gray, kernel, 2)
    gray = cv2.erode(gray, kernel, 2)

cv2.imshow('jkhufsdeuhkivfdqs', gray)
cv2.waitKey(0)

minLineLength = 200
maxLineGap = 15
lines = cv2.HoughLinesP(gray, 0.05, np.pi / 180, 10, 0, 100)  # ,minLineLength,maxLineGap)

# print lines
imgs = gray * 0

for (x1, x2, y1, y2) in lines[0]:
    cv2.line(imgs, (x1, x2), (y1, y2), (255, 255, 255), 1)

# cv2.imshow('haha',imgs)
# imgs = gray
print 'Hough termine'
cv2.imwrite('./houghlines5.jpg', imgs)
cv2.imshow('jkhufsdeuhkivfdqs', imgs)
cv2.waitKey(0)
imgs = gray
h, l = imgs.shape

# # # # # # # # # # # # # # # #
sumh = np.zeros(l, np.uint8)
sumg = np.zeros(h, np.uint8)
for y in range(0, h):
    for x in range(0, l):
        if imgs[y, x] > 126:
            sumg[y] += 1
            sumh[x] += 1

print 'sum termine'

# # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # #


# sumt = sumh
#
# seuil = np.max(sumt)*0.9# * 1/2
# dist = distances_hb(seuil, sumt)
# cl = classific(dist)
# fr = frontiere(cl)
# # IdSquare = whoIsSquare(cl)
# # sizes = getPosition(sumt, seuil, IdSquare)
#
#
#
# print 'seuil'
# print seuil
#
# print dist
# print cl
# print fr
#
#
# # print 'largeur moyen du carre :'
# # print IdSquare
#
# # print 'position des carres'
# # print sizes
#
#
#
# plotseuil = []
# for i in sumt:
#     plotseuil.append(seuil)
#
# plt.plot(sumt)
# plt.plot(plotseuil)
# # plt.plot(conv)
# plt.ylabel('some numbers')
# plt.show()
# cv2.waitKey(0)
# np.transpose(sumt)
# sumt.tofile('foo.csv', sep='\n', format='%10.1f')
# cv2.imwrite('./houghlines5.jpg', imgs)
# cv2.waitKey(0)

# faiblesse : le seuil (variable seuil) => definir une heuristigue
#             eventuellement, il y aurait aussi la variable echantillonage





seuilh = np.max(sumh) * 0.2

plotseuilh = []
for i in sumh:
    plotseuilh.append(seuilh)
plt.plot(sumh)
plt.plot(plotseuilh)
plt.ylabel('sumH')
plt.show()

disth = distances_hb(seuilh, sumh)
clh = classific(disth)
IdSquareh = whoIsSquare(clh)
sizesh = getPosition(sumh, seuilh, IdSquareh)

plotseuilh = []
for i in sumh:
    plotseuilh.append(seuilh)

plt.plot(sumh)
plt.plot(plotseuilh)
plt.ylabel('sumH')
plt.show()


seuilg = np.max(sumg) * 0.2
distg = distances_hb(seuilg, sumg)
clg = classific(distg)
IdSquareg = whoIsSquare(clg)
sizesg = getPosition(sumg, seuilg, IdSquareg)

plotseuilG = []
for i in sumg:
    plotseuilG.append(seuilg)

plt.plot(sumg)
plt.plot(plotseuilG)
plt.ylabel('sumG')
plt.show()

shape = getSquareShape(sizesg, sizesh)


print shape

for (pt1, pt2) in shape:
    cv2.rectangle(origine, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 0, 0))

cv2.imwrite('end2.png', origine)

print 'classes hautes et gauche'
print clh
print clg

print 'classes selectionner (H, G)'
print IdSquareh
print IdSquareg
