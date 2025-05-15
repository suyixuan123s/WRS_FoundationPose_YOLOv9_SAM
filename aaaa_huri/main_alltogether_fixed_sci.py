from commonimport import *
import tubepuzzlefaster_sci as tp

if __name__ == '__main__':

    yhx = robothelper.RobotHelperX(usereal=True)
    yhx.movetox(yhx.rbt.initrgtjnts, armname="rgt")
    yhx.movetox(yhx.rbt.initlftjnts, armname="lft")
    yhx.closegripperx(armname="rgt")
    yhx.closegripperx(armname="lft")
    yhx.opengripperx(armname="rgt")
    yhx.opengripperx(armname="lft")
    lctr = locfixed.LocatorFixed(homomatfilename="rightfixture_homomat1")
    # lctr = loc.Locator(standtype = "light")

    armname = "rgt"
    ppplanner_tb = ppp.PickPlacePlanner(lctr.tubebigcm, yhx)
    ppplanner_ts = ppp.PickPlacePlanner(lctr.tubesmallcm, yhx)

    objpcd = lctr.capturecorrectedpcd(yhx.pxc)
    pcdnp = p3dh.genpointcloudnodepath(objpcd, pntsize=5)
    # pcdnp.reparentTo(yhx.base.render)
    elearray, eleconfidencearray = lctr.findtubes(lctr.tubestandhomomat, objpcd, toggledebug=False)


    import registration.pattern as ptn
    pto = ptn.Pattern(root=".")
    # tmpelearray = np.array([[0,0,1,1,0,0,2,2,2,2],
    #                         [1,1,0,0,1,1,0,0,0,0],
    #                         [0,0,0,1,0,0,0,0,0,0],
    #                         [0,0,0,0,0,0,0,0,1,0],
    #                         [0,0,0,0,0,0,0,0,0,0]])
    pto.setpattern(elearray)
    # pto.gencad(homomat=lctr.tubestandhomomat).reparentTo(base.render)
    # elearray_ext = np.zeros((state.shape[0]*2, state.shape[1]))
    # elearray_ext[0:state.shape[0], 0:state.shape[1]] = state[:, :]
    # yhx.p3dh.genframe(pos=lctr.tubestandhomomat[:3, 3], rotmat=lctr.tubestandhomomat[:3, :3]).reparentTo(yhx.base.render)
    tubestandcm = lctr.gentubestand(homomat=lctr.tubestandhomomat)
    tubestandcm.reparentTo(base.render)
    # base.run()

    # goal_pattern = np.array([[2,2,2,0,0,0,0,1,1,1],
    #                          [2,2,2,0,0,0,0,1,1,1],
    #                          [2,2,2,0,0,0,0,1,1,1],
    #                          [2,2,2,0,0,0,0,1,1,1],
    #                          [2,2,2,0,0,0,0,1,1,1]])
    tpobj = tp.TubePuzzle(elearray, goalpattern=None)
    if tpobj.isdone(tp.Node(elearray)):
        print("Tubes are arranged!")
        sys.exit()
    # weight_array = np.zeros_like(state)
    # weight_array[0,0] = 1
    # weight_array[1,0] = 1
    # path = tpobj.atarSearch(weight_array)
    path = tpobj.atarSearch()
    for node in path:
        print(node)
    #

    # counter = [0]
    # tubemnplist = [[]]
    # tubestandhomomat = [lctr.tubestandhomomat]
    # def update(path, counter, lctr, tubemnplist, task):
    #     if counter[0] < len(path):
    #         # lctr.gentubestand(homomat=tubestandhomomat[0]).reparentTo(base.render)
    #         state = path[counter[0]]
    #         tubecmlist = lctr.gentubes(state.grid, tubestandhomomat[0], alpha=1)
    #         for tubenp in tubemnplist[0]:
    #             tubenp.detachNode()
    #         tubemnplist[0] = []
    #         for tubecm in tubecmlist:
    #             tubemnplist[0].append(tubecm)
    #             tubecm.reparentTo(base.render)
    #         if base.inputmgr.keymap['space'] is True:
    #             base.inputmgr.keymap['space'] = False
    #             counter[0] += 1
    #     # else:
    #     #     counter[0] = 0
    #     return task.again
    #
    # taskMgr.doMethodLater(0.05, update, "update",
    #                       extraArgs=[path, counter, lctr, tubemnplist],
    #                       appendTask=True)
    #
    # base.run()

    numikmsmpall = []
    jawwidthmsmpall = []
    objmsmpall = []
    othersmsmpall = []
    lastrgtarmjnts = yhx.rbt.initrgtjnts
    lastlftarmjnts = yhx.rbt.initlftjnts
    i = 0
    # for bad state
    badarraylist = []
    badstartgoals = []
    finalpath = [path[0]]
    counter = 0
    while i < len(path) - 1:
        print(i, len(path) - 1)
        # try:
        nodepresent = path[i]
        nodenext = path[i + 1]
        finalpath.append(nodenext)
        griddiff = nodepresent.grid - nodenext.grid
        initij = np.where(griddiff > 0)
        initij = (initij[0][0], initij[1][0])
        tubeid = int(nodepresent.grid[initij[0], initij[1]])
        ppplanner = ppplanner_tb if tubeid == 1 else ppplanner_ts
        goalij = np.where(griddiff < 0)
        goalij = (goalij[0][0], goalij[1][0])
        collisionelearray = copy.deepcopy(nodepresent.grid)
        collisionelearray[initij[0], initij[1]] = 0
        renderingelearray = copy.deepcopy(nodepresent.grid)
        # renderingelearray[goalij[0], goalij[1]] = nodenext.grid[goalij[0], goalij[1]]
        # print(collisionelearray)
        # print(renderingelearray)
        print(nodepresent.grid)
        print(nodepresent.grid-nodenext.grid)
        collisiontbcmlist = lctr.gentubes(collisionelearray, tubestand_homomat=lctr.tubestandhomomat)

        initpos_normalized = np.array(
            [lctr.tubeholecenters[initij[0], initij[1]][0], lctr.tubeholecenters[initij[0], initij[1]][1], 5])
        initpos = rm.homotransformpoint(lctr.tubestandhomomat, initpos_normalized)
        inithm = copy.deepcopy(lctr.tubestandhomomat)
        inithm[:3, 3] = initpos
        goalpos_normalized = np.array(
            [lctr.tubeholecenters[goalij[0], goalij[1]][0], lctr.tubeholecenters[goalij[0], goalij[1]][1], 5])
        goalpos = rm.homotransformpoint(lctr.tubestandhomomat, goalpos_normalized)
        goalhm = copy.deepcopy(lctr.tubestandhomomat)
        goalhm[:3, 3] = goalpos

        obscmlist = yhx.obscmlist + [tubestandcm] + collisiontbcmlist
        td = False
        # if len(path) == 8:
        #     counter += 1
        #     if counter == 2:
        #         # pto.setpattern(nodepresent.grid)
        #         pto.setpattern(nodenext.grid)
        #         # newgrid = (nodenext.grid+nodepresent.grid)/2
        #         # griddiff = nodenext.grid-nodepresent.grid
        #         # newgrid[griddiff!=0] = 0
        #         # pto.setpattern(newgrid)
        #         pto.gencad(homomat=lctr.tubestandhomomat).reparentTo(base.render)
        #         td = True
        numikmsmp, jawwidthmsmp, objmsmp = ppplanner.findppmotion_symmetric(inithm, goalhm, armname=armname,
                                                                            rbtinitarmjnts = [lastrgtarmjnts, lastlftarmjnts],
                                                                            finalstate="uo",
                                                                            obscmlist=obscmlist, userrt=True,
                                                                            primitivedistance_init_foward=130,
                                                                            premitivedistance_init_backward=130,
                                                                            primitivedistance_final_foward=130,
                                                                            premitivedistance_final_backward=130,
                                                                            toggledebug = td)
        ## for toggle_debug, check the collisions between the hand and the tubes
        # for tbcm in collisiontbcmlist:
        #     tbcm.reparentTo(yhx.base.render)
        #     tbcm.setColor(1,0,0,.2)
        #     tbcm.showcn()
        # yhx.base.run()
        if numikmsmp is None:
            finalpath.pop(-1)
            tmpelearray = copy.deepcopy(nodepresent.grid)
            badarraylist.append(tmpelearray)
            badstartgoals.append([[initij[0], initij[1]], [goalij[0], goalij[1]]])
            tpobj = tp.TubePuzzle(tmpelearray)
            weightarray = np.zeros_like(nodepresent.grid)
            # weight_array[0,0] = 1
            # weight_array[1,0] = 1
            for id, onebadarray in enumerate(badarraylist):
                if np.array_equal(onebadarray, tmpelearray):
                    weightarray[badstartgoals[id][0][0], badstartgoals[id][0][1]] = id+1
                    weightarray[badstartgoals[id][1][0], badstartgoals[id][1][1]] = id+1
            path = tpobj.atarSearch(weightarray)
            for node in path:
                print(node)
            finalpath[-1] = path[0]
            i = 0
            # print("Failure")
            # print(badarraylist)
            # print(weight_array)
            continue
        else:
            i += 1
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
                if tubeid == 1:
                    tmpobjcm.setColor(1, 1, 0, 1)
                else:
                    tmpobjcm.setColor(1, 0, 1, 1)
                objcmmp.append(tmpobjcm)
            objcmmsmp.append(objcmmp)
        objmsmpall += objcmmsmp
        othersmsmp = []
        for idms in range(len(numikmsmp)):
            othersmp = []
            for idmp in range(len(numikmsmp[idms])):
                renderingtbcmlist = lctr.gentubes(renderingelearray, tubestand_homomat=lctr.tubestandhomomat)
                othersmp.append([tubestandcm] + renderingtbcmlist)
            othersmsmp.append(othersmp)
        othersmsmpall += othersmsmp
    anime.animationgen(yhx, numikmsmpall, jawwidthmsmpall, objmsmpall, othersmsmpall)
    yhx.base.run()

    counter = [0]
    tubemnplist = [[]]
    tubestandhomomat = [lctr.tubestandhomomat]
    def update(path, counter, lctr, tubemnplist, task):
        if counter[0] < len(path):
            for np in tubemnplist[0]:
                np.detachNode()
            tubemnplist[0] = []
            state = path[counter[0]]
            pto.setpattern(state.grid)
            np = pto.gencad(homomat=lctr.tubestandhomomat)
            tubemnplist[0].append(np)
            np.reparentTo(base.render)
            if base.inputmgr.keymap['space'] is True:
                base.inputmgr.keymap['space'] = False
                counter[0] += 1
        else:
            counter[0] = 0
        return task.again

    taskMgr.doMethodLater(0.05, update, "update",
                          extraArgs=[finalpath, counter, lctr, tubemnplist],
                          appendTask=True)

    base.run()
