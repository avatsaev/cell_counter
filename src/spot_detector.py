import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt


# la fonction "exeCalc(dilated)" execute le scripte et renvoi la positions des carres et la moyenne de la taille des carres detecte.
# [ [ positions des carres ], moyenne des carres ]
#         => avec [positions des carres ] = [ [point sup gauche, point inf droit], [point sup ...], ... ]
#
# on peut executer le script par console avec les parametres suivants :
#     - i : path de l'image a detecte (prendre l'image dilate de contour_pattern.py
#     - p : path de sortie de l'image contenant les carre (defaut, le dossier courrant)
#     - s : path de l'image sur la-quelle on souhaite dessiner les carre (defaut, une image noir)


def openImg(gray):
    print("SPORT DETECTOR: openImg...")
    "diminue le bruit de l'image en praticant quelques ouvertures"
    kernel = np.ones((5, 5), np.uint8)
    for i in range(0, 5):
        gray = cv2.dilate(gray, kernel, 5)
        gray = cv2.erode(gray, kernel, 5)
    return gray

def getHoughImg(gray):
    print("SPORT DETECTOR: getHoughImg...")


    #########
    #TODO: This function is the biggest bottleneck, needs to be optimised (maybe by distributing the workload on several CPU cores)
    #########

    "renvoie un image en NG qui contient les contour des carres determine par la transformee de Hough"
    lines = cv2.HoughLinesP(gray, 0.05, np.pi / 180, 100, 0, 75)

    imgs = gray * 0

    # for (x1, x2, y1, y2) in lines[0]:
    #     cv2.line(imgs, (x1, x2), (y1, y2), (255, 255, 255), 1)

    for i in lines:
        cv2.line(imgs, (i[0, 0], i[0, 1]), (i[0, 2], i[0, 3]), (255, 255, 255), 1)



    return imgs


def sumHandG(gethough):


    print("SPORT DETECTOR: sumHandG...")
    "renvoie le nombre de pixels blancs par colonne, puis par ligne de pixel [[nbr de pix/colonne],[nbr de pix/ligne]]"

    h, l = gethough.shape
    sumg = np.zeros(h, np.uint8)
    sumh = np.zeros(l, np.uint8)
    for y in range(0, h):
        for x in range(0, l):
            if gethough[y, x] > 126:
                sumg[y] += 1
                sumh[x] += 1


    return [sumh, sumg]


def distances_hb(seuil, f):
    print("SPORT DETECTOR: distances_hb...")
    "renvoie la distance entre chaque pics. le seuil determine a partir de quel valeur on a un \"pics\" [ periode, periode, ...]"
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


def quadra(m):
    print("SPORT DETECTOR: quadra..")
    "moyenne quadratique"
    a = m
    np.power(a, 2)
    summ = sum(a)
    summ = summ * 1 / len(a)
    return np.sqrt(summ)


# classification barbar

def classific(dist):
    print("SPORT DETECTOR: classific...")
    "renvoi les centres de gravite des classes, et leur nombre d'elements [[centre, nbr element assoier a ce centre], [...], ...]"
    sdist = dist
    cl = [[]]  # ensemble de classes
    echantillonage = 2 # defini la severiter dans le choix de classification ( + c'est petit, + on genere de class, + on est grand, + on est sensible au bruit)
                        # ==> ici, si echantillonage = 10, on classe toutes les valeurs par multiple de 10,
                        #ex : [2,10,5,34,6,5,11,35,19] devient [[2,6,5],[10,11,19], [], [34, 35]]

    for i in sdist:
        while np.size(cl, 0) <= int(round(i / echantillonage, 0)):
            cl.append([])
        c = cl[int(round(i / echantillonage, 0))]
        c.append(i)
        cl[int(round(i / echantillonage, 0))] = c



     # definition des classes principales, [[nmbre d'elments],[multiple d'echantillonage inferieur],[multiple d'echantillonage superieur]]
            # ==> on rassemble toutes les classes qui ne sont pas separees par []
            # si on reprend notre exemple ex : [[2,6,5],[10,11,19], [], [34, 35]]        devient [[6, [0,1]], [2, [3,3]]
            #     pour les multiples de 10 :   [   0        1        2      3   ] on rassemble :  [0 et 1]      [3]
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

    # print 'taille des classes [[nmbre d\'elments],[multiple d'echantillonage inferieur],[multiple d'echantillonage superieur]]'
    # print coord

    g = [] #calcul des centres de gravites ou moyenne des periodes classee (c'est pareil =P)
    for clas in coord:
        g.append([0, clas[0]])
        for i in range(clas[1], clas[2]+1):
            pos = len(g)-1
            g[pos][0] += sum(cl[i])
        g[pos][0] /= clas[0]

    return g #[[centre de gravite de la classe, cardinal de la classe]


def frontiere(g):
    print("SPORT DETECTOR: frontiere...")
    "definie les frontiere entre les centres de gravite. [val frontiere entre la classe 1 et 2, val frontiere entre la classe 2 et 3, ... ]"
                # ==>  g = [centre graviter, valeur sans importance]
    fr = []
    for i in range(0,len(g)-1):
        fr.append((g[i+1][0]+g[i][0])/2)
    return fr


def whoIsSquare(classifics):
    print("SPORT DETECTOR: whoIsSquare...")
    "renvoi la classe qui est la plus suceptible de representer l'ensemble des largeurs de carres [centre gravite, cardinal de la classe]"
    toClass = list(classifics)
    c = [0, [0, 0]] # [val,[frontieres de classifictations]]
    sort = [[0,0,0], [0,0,0], [0,0,0]] # [val, nbrElements, position dans la classification] On s'atend a ce que 3 classes sortes du lots]
    ireset = 0
    for s in sort:                          # on prends les trois classes qui reviennent le plus regulierement (le # le + elevee)
        i = ireset
        for [val, nbrEl] in toClass:
            if s[1] <= nbrEl and nbrEl > 4: # on considere qu'une classe qui ne possede pas au moins 5 elements n'est pas interressante
                s[1] = nbrEl
                s[0] = val
                s[2] = i
            i += 1
        if len(toClass) > 1:
            toClass.pop(s[2]-ireset)
            ireset += 1

    # on obtient 3 classes : [[classe largeur contour], [classe largeur carre], [classe distance entre 2 carres]]
    #               La plus recurente contient principalement la largeur du contour des carres, mais aussi un peut de bruit.
    #               les deux suivantes contiennent la largeur de nos carres et la distance entre 2 carres
    #                   ==> les carres etants plus larges que la distance entre 2 specimens, on choisi de prendre la classe avec
    #                       le centre de gravitee le plus elevee parmis nos trois candidats.

    f = frontiere(classifics) # on obtient les valeurs qui separe les classes

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

    return c # [ centre de gravitee de la classe largeur carre, [frontiere inferieur de la classe, frontiere superier de la classe]]

def getPosition(sumHorG, seuil, whoSquare):
    print("SPORT DETECTOR: getPosition..")
    "renvoi les positions de debut et de fin des carres de l'image. [[x1,x2],[x1,x2],...]"
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
                square.append([start, i])
            start = i
            length = 1
            temp = 1
        elif (axe[i] <= seuil and temp != 2):
            if target[0] < length < target[1]:
                square.append([start, i])
            start = i
            length = 1
            temp = 2
        else:
            length += 1

    return square

def getSquareShape(getposition):
    print("SPORT DETECTOR: getSquareShape...")
    "renvoi la position des carres, [ [[point sup gauche],[point inf droit]], ...]"
    shape = []

    for (x1, x2) in getposition[0]:
        for(y1, y2) in getposition[1]:
            shape.append([[x1, y1], [x2, y2]])

    return shape

def drawRect(img, getsquareshape):
    print("SPORT DETECTOR: drawRect...")
    "dessine en vert les rectangles contenus de getsquareshape. dans img \n\t ==>  getsquareshape = [[pt1, pt2], [pt1,pt2] ...]"

    for (pt1, pt2) in getsquareshape:
        cv2.rectangle(img, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 0))
    return img

# kernel = np.ones((5,5),np.uint8)
# gray = cv2.dilate(edge,kernel,2)


########################################################################################################################
########################################################################################################################
########################################################################################################################

# gray = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output/dilated.png', cv2.IMREAD_GRAYSCALE)
# origine = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/output/opening_morph.png')
#     dilated =  cv2.imread('dilated', cv2.IMREAD_GRAYSCALE)
#     origine = cv2.imread('B:/Projets/Python/cell_counter/tests/contours/contours_sample_3_raw.jpg')

def exeCalc(dilated):
    print("SPORT DETECTOR: exeCalc...")
    "execute le scripte de detection des carres sur l'image de sortie de contour_pattern.py, renvoie [[position des carres],[taille moyenne]]"

    dilated = openImg(dilated)   # on filtre un peut l'image avec des ouvertures
    imgs = getHoughImg(dilated)  # on retire les principales lignes droites de notre image. on souhaite obtenir des portions majeur de quelques carres
    sumXY = sumHandG(imgs)  # on somme tous les pixel suivant les colonnes, puis les lignes.
                            # ==> on souhaite obtenir des "pics" aux positions des limites des carres
    i = 0
    seuil = [[], []]
    dist = [[], []]
    cl = [[], []]
    IdSquare = [[], []]
    sizes = [[], []]

    for sumt in sumXY: #on fait ce travail pour les sommes suivant les colonnes, puis les lignes.
        seuil[i] = np.max(sumt) * 4/6 # on etablie de maniere empirique le seuil audessus duquel on considere un valeur comme etant un pic. !! trop bas, on a trop de bruit, trop haut, on rate trop de pics !!
        dist[i] = distances_hb(seuil[i], sumt)  # on calcul la distance entre chaque pics. on souhaite en retirer une periode qui correspondrait a la largeur d'un carre
        cl[i] = classific(dist[i])              # on classifie les diferentes periodes trouvees
        IdSquare[i] = whoIsSquare(cl[i])        # on cherche parmis nos classes la-quelle est la plus probable de representer la largeur de nos carres : renvoi [largeur moyen, [frontiere inf, frontiere sup]]
        sizes[i] = getPosition(sumt, seuil[i], IdSquare[i]) # on recalcul les periodes sur la somme/colonne, et on enregistre la position des periodes correspondants a la classe "largeur carre"
        i += 1

        # plotseuil = [[], []]    # partie test #######################################
        # for blblblbl in sumt:
        #     plotseuil[i-1].append(seuil[i-1])
        #
        # plt.plot(sumt)
        # plt.plot(plotseuil[i-1])
        # plt.ylabel('sum')
        # plt.show()
        #
        # print 'classes hautes et gauche'
        # print cl[i-1]
        #
        # print 'classes selectionner (H, G)'
        # print IdSquare[i-1]


    shape = getSquareShape(sizes) # on combine la position x et y des carres pour identifier leurs positions
    mean = max(IdSquare[0][0], IdSquare[1][0])

    #distance between the sqares
    interspot_dist=75

    return [shape, mean, interspot_dist]  # [[ [point sup gauche, point inf droit], [point sup ...], ... ], moyenne de la taille de tous les carres]

def printSquare(shape, support, path="output/"):
    print(shape[0])
    print("SPORT DETECTOR: printSquare...")
    "dessine les carres et les dessines sur le support, et l'enregistre dans le Path, ou a coter (default)"
    support = drawRect(support, shape)  # on dessine les carres trouve
    cv2.imwrite(str(path) + '/squares.png', support)  # on enregistre l'image

    return support
# la taille des carres est contenus dans IdSquare : [ [largeur suivant X, [] ], [largeur suivant Y, [] ] ]
    # ==> La largeur suivant X et Y peut etre diferent !!! privilegiez la plus grande valeur !!
        # ==> soit largeur des carre = max( IdSquare[0, 0], IdSquare[1, 0])


# faiblesse : le seuil (variable seuil) => definir une heuristigue
#             eventuellement, il y aurait aussi la variable echantillonage

########################################################################################################################
########################################################################################################################
########################################################################################################################
#
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,
#     help = "Path to the query image")
#
# ap.add_argument("-p", "--path", required = False,
#     help = "Path to print the finding square (default : print in current folder)")
#
# ap.add_argument("-s", "--support", required = False,
#     help = "Path to the image for print finding square (default : with -e, print square with black screen)")
#
#
# args = vars(ap.parse_args())
#
# img = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
# (shape, mean) = exeCalc(img)
#
# if args['support'] is None:
#     support = img * 0
# else:
#     support = cv2.imread(args['support'])
#
# printSquare(shape, support, args['path'])
