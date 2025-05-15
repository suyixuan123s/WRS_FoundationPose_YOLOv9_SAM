from motion import smoother as sm
from motion import checker as ctcb
from motion import collisioncheckerball as cdck
from motion.rrt import rrtconnect as rrtc
import tro.tro_robothelper as robothelper
from pandaplotutils import pandactrl as pc
import numpy as np
import utiltools.robotmath as rm
import tro.tro_animationgenerator as anime
import os
import copy
import tubepuzzlefaster as tp
import tubepuzzle_newstand as tp_nst
import tro.tro_locator as loc
import locatorfixed_newstand as locfixed_nst
import cv2
import environment.collisionmodel as cm
import utiltools.thirdparty.p3dhelper as p3dh
import utiltools.thirdparty.o3dhelper as o3dh
import pickle
import tro.tro_pickplaceplanner as ppp

if __name__ == '__main__':

    yhx = robothelper.RobotHelperX(usereal=True)
    yhx.movetox(yhx.robot_s.initrgtjnts, armname="rgt")
    yhx.movetox(yhx.robot_s.initlftjnts, armname="lft")
    yhx.closegripperx(armname="rgt")
    yhx.closegripperx(armname="lft")
    yhx.opengripperx(armname="rgt")
    yhx.opengripperx(armname="lft")
    lctr = loc.TLocator(directory='..')

    armname = "rgt"
    ppplanner = ppp.ADPlanner(lctr.srccm, yhx)

    tgtpcd = lctr.capturecorrectedpcd(yhx.pxc)
    objhomomat = lctr.findobj(tgtpcd)
    objcm = lctr.genobjcm(homomat=objhomomat)

    inithomomat = copy.deepcopy(objhomomat)
    goalhomomat = copy.deepcopy(inithomomat)

    numikmsmpall = []
    jawwidthmsmpall = []
    objmsmpall = []
    othersmsmpall = []
    lastrgtarmjnts = yhx.robot_s.initrgtjnts
    lastlftarmjnts = yhx.robot_s.initlftjnts

    obscmlist = yhx.obscmlist
    numikmsmp, jawwidthmsmp, objmsmp = ppplanner.findppmotion(inithomomat, goalhomomat, armname=armname,
                                                                rbtinitarmjnts = [lastrgtarmjnts, lastlftarmjnts],
                                                                finalstate="uo", obscmlist=obscmlist, userrt=True, toggledebug=False)
    if armname is "rgt":
        lastrgtarmjnts = numikmsmp[-1][-1][1]
        lastlftarmjnts = lastlftarmjnts
    else:
        lastrgtarmjnts = lastrgtarmjnts
        lastlftarmjnts = numikmsmp[-1][-1][2]
    numikmsmpall += numikmsmp
    jawwidthmsmpall += jawwidthmsmp
    objcmmsmp = []
    for objhomomatmp in objmsmp:
        objcmmp = []
        for objhomomat in objhomomatmp:
            tmpobjcm = copy.deepcopy(ppplanner.objcm)
            tmpobjcm.set_homomat(objhomomat)
            tmpobjcm.setColor(.5, .5, .5, 1)
            objcmmp.append(tmpobjcm)
        objcmmsmp.append(objcmmp)
    objmsmpall += objcmmsmp
    othersmsmp = []
    for idms in range(len(numikmsmp)):
        othersmp = []
        for idmp in range(len(numikmsmp[idms])):
            othersmp.append([objcm])
        othersmsmp.append(othersmp)
    othersmsmpall += othersmsmp
    anime.animationgen(yhx, numikmsmpall, jawwidthmsmpall, objmsmpall, othersmsmpall)

    yhx.base.run()
    # counter = [0]
    # tubemnplist = [[]]
    # tubestandhomomat = [homomat]
    # def update(path, counter, lctr, task):
    #     if counter[0] < len(path):
    #         lctr.showTubestand(homomat=tubestandhomomat[0])
    #         state = path[counter[0]]
    #         lctr.showTubes(state.grid, tubestandhomomat[0])
    #         if base.inputmgr.keyMap['space'] is True:
    #             base.inputmgr.keyMap['space'] = False
    #             counter[0] += 1
    #     # else:
    #     #     counter[0] = 0
    #     return task.again
    #
    # taskMgr.doMethodLater(0.05, update, "update",
    #                       extraArgs=[path, counter, lctr],
    #                       appendTask=True)
