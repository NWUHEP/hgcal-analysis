import ROOT
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from root_pandas import read_root
import operator


class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.

    @property
    def r(self):
        return np.sqrt((self.x)**2 + (self.y)**2)

    @property
    def eta(self):
        theta = np.arccos(self.z/self.r)
        return -np.log(np.tan(theta/2.))

    @property
    def phi(self):
        return np.arctan2(self.y, self.x)

class Cell:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.wafer = 0
        self.c     = 0
        self.center = Position()
        self.corners = [Position(), Position(), Position(), Position()]

    def box(self):
        return ROOT.TBox(self.corners[0].x, self.corners[0].y, \
                         self.corners[2].x, self.corners[2].y)

    def __eq__(self, other):
        return self.id==other.id

class TriggerCell:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.wafer = 0
        self.tc    = 0
        self.center = Position()
        self.cells = []

class Module:
    def __init__(self):
        self.id = 0
        self.zside = 0
        self.layer = 0
        self.module = 0
        self.center = Position()
        self.tcells = []

class Motherboard:
    def __init__(self):
        self.zside = 0
        self.layer = 0
        self.center = Position()
        self.modules = []
        
def float_equal(x1, x2):
    prec = 1.e-4
    if abs(x1)<prec and abs(x2)>prec: return False
    elif abs(x1)<prec and abs(x2)<prec: return True
    else: return abs( (x1-x2)/x1)<prec

def compare_lines(line1, line2):
    xy11 = (line1.GetX1(), line1.GetY1())
    xy12 = (line1.GetX2(), line1.GetY2())
    xy21 = (line2.GetX1(), line2.GetY1())
    xy22 = (line2.GetX2(), line2.GetY2())
    samecorner1 = (float_equal(xy11[0],xy21[0]) and float_equal(xy11[1],xy21[1])) or (float_equal(xy11[0],xy22[0]) and float_equal(xy11[1],xy22[1]))
    samecorner2 = (float_equal(xy12[0],xy21[0]) and float_equal(xy12[1],xy21[1])) or (float_equal(xy12[0],xy22[0]) and float_equal(xy12[1],xy22[1]))
    #print "[",xy11,xy12,"]","[",xy21,xy22,"]",(samecorner1 and samecorner2)
    return samecorner1 and samecorner2

def boxlines(box):
    lines = []
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX1(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY1(), box.GetX2(), box.GetY1()))
    lines.append(ROOT.TLine(box.GetX1(), box.GetY2(), box.GetX2(), box.GetY2()))
    lines.append(ROOT.TLine(box.GetX2(), box.GetY1(), box.GetX2(), box.GetY2()))
    return lines



def loadtriggermapping(layer,zside,inputFileName,sdet=None,
                       outputFileName = None, 
                       triggeronly    = None):
    inputFile = ROOT.TFile.Open(inputFileName)

    treeModules      = inputFile.Get("hgcaltriggergeomtester/TreeModules")
    treeTriggerCells = inputFile.Get("hgcaltriggergeomtester/TreeTriggerCells")
    treeCells        = inputFile.Get("hgcaltriggergeomtester/TreeCells")

    treeModules.__class__      = ROOT.TTree
    treeTriggerCells.__class__ = ROOT.TTree
    treeCells.__class__        = ROOT.TTree
    
    
    
    ## filling cell map
    cells = {}
    if sdet is None:
        cut = "layer=={0} && zside=={1}".format(layer,zside)
    else:
        cut = "layer=={0} && zside=={1} && subdet=={2}".format(layer,zside,sdet)
    treeCells.Draw(">>elist1",cut , "entrylist")
    entryList1 = ROOT.gDirectory.Get("elist1")
    entryList1.__class__ = ROOT.TEntryList
    nentry = entryList1.GetN()
    treeCells.SetEntryList(entryList1)
    for ie in range(nentry):
        if ie%10000==0: print("Entry {0}/{1}".format(ie, nentry))
        entry = entryList1.GetEntry(ie)
        treeCells.GetEntry(entry)
        cell = Cell()
        cell.id       = treeCells.id
        cell.zside    = treeCells.zside
        cell.layer    = treeCells.layer
        cell.wafer    = treeCells.wafer
        cell.c        = treeCells.cell

        cell.center.x = treeCells.x
        cell.center.y = treeCells.y
        cell.center.z = treeCells.z
        cell.corners[0].x = treeCells.x1
        cell.corners[0].y = treeCells.y1
        cell.corners[1].x = treeCells.x2
        cell.corners[1].y = treeCells.y2
        cell.corners[2].x = treeCells.x3
        cell.corners[2].y = treeCells.y3
        cell.corners[3].x = treeCells.x4
        cell.corners[3].y = treeCells.y4
        if cell.id not in cells: cells[cell.id] = cell

    ## filling trigger cell map
    triggercells = {}
    treeTriggerCells.Draw(">>elist2", cut, "entrylist")
    entryList2 = ROOT.gDirectory.Get("elist2")
    entryList2.__class__ = ROOT.TEntryList
    nentry = entryList2.GetN()
    treeTriggerCells.SetEntryList(entryList2)
    for ie in range(nentry):
        if ie%10000==0: print("Entry {0}/{1}".format(ie, nentry))
        entry = entryList2.GetEntry(ie)
        treeTriggerCells.GetEntry(entry)
        triggercell = TriggerCell()
        triggercell.id       = treeTriggerCells.id
        triggercell.zside    = treeTriggerCells.zside
        triggercell.layer    = treeTriggerCells.layer
        triggercell.wafer    = treeTriggerCells.wafer
        triggercell.tc       = treeTriggerCells.triggercell
    
        triggercell.center.x = treeTriggerCells.x
        triggercell.center.y = treeTriggerCells.y
        triggercell.center.z = treeTriggerCells.z
        for cellid in treeTriggerCells.c_id:
            if not cellid in cells: raise Exception("Cannot find cell {0} in trigger cell".format(cellid))
            cell = cells[cellid]
            triggercell.cells.append(cell)
        triggercells[triggercell.id] = triggercell
    
    # filling module map
    modules = {}
    treeModules.Draw(">>elist3", cut, "entrylist")
    entryList3 = ROOT.gDirectory.Get("elist3")
    entryList3.__class__ = ROOT.TEntryList
    nentry = entryList3.GetN()
    treeModules.SetEntryList(entryList3)
    for ie in range(nentry):
        if ie%10000==0: print("Entry {0}/{1}".format(ie, nentry))
        entry = entryList3.GetEntry(ie)
        treeModules.GetEntry(entry)
        module = Module()
        module.id       = treeModules.id
        module.zside    = treeModules.zside
        module.layer    = treeModules.layer
        module.module   = treeModules.module
        module.center.x = treeModules.x
        module.center.y = treeModules.y
        module.center.z = treeModules.z
        for tcellid in treeModules.tc_id:
            if not tcellid in triggercells: raise Exception("Cannot find trigger cell {0} in module".format(tcellid))
            tcell = triggercells[tcellid]
            module.tcells.append(tcell)
        modules[module.id] = module

    # filling motherboard map
    motherboards = {}
    motherboard = Motherboard()
    for i, mod in enumerate(sorted(modules.values(), key=operator.attrgetter('center.phi'))):
        motherboard.modules.append(mod)
        #if (((i+1)%5 == 0) or i == len(modules)-1):
        if ((i+1)%4 == 0):
            motherboard.id = len(motherboards)
            motherboard.zside = mod.zside
            motherboard.layer = mod.layer
            motherboard.center.x = np.mean([thismod.center.x for thismod in motherboard.modules])
            motherboard.center.y = np.mean([thismod.center.y for thismod in motherboard.modules])
            motherboard.center.z = np.mean([thismod.center.z for thismod in motherboard.modules])
            motherboards[motherboard.id] = motherboard
            motherboard = Motherboard()
            
    
    print("===========================================")
    print("Read", len(cells), "cells" )
    print("Read", len(triggercells), "trigger cells")
    print("Read", len(modules), "modules")
    print("Read", len(motherboards), "motherboards")
    print("===========================================")
    
    flatlist = []
    # filling list
    ######################################################
    if triggeronly is None:
        labels = ["c_wafer","c_x","c_y","c_z","c_c",\
                  "tc_id","tc_wafer","tc_x","tc_y","tc_z","tc_tc",\
                  "mod_id","mod_x","mod_y","mod_z","mod_mod",\
                  "mboard_id", "mboard_x", "mboard_y", "mboard_z"]
   
        #for id,module in modules.items():
        for id,motherboard in motherboards.items():
            for module in motherboard.modules:
                for triggercell in module.tcells:
                    for cell in triggercell.cells:
                        row =  [cell.wafer,cell.center.x,cell.center.y,cell.center.z,cell.c,\
                                triggercell.id,triggercell.wafer,triggercell.center.x,triggercell.center.y,\
                                triggercell.center.z,triggercell.tc,\
                                module.id,module.center.x,module.center.y,module.center.z,module.module,\
                                motherboard.id, motherboard.center.x, motherboard.center.y, motherboard.center.z]
                        flatlist.append(row)

    else:
        labels = ["c_wafer","c_x","c_y","c_z","c_c",\
                  "tc_id","tc_wafer","tc_x","tc_y","tc_z","tc_tc"]
    
        for id,triggercell in triggercells.items():
            for cell in triggercell.cells:
                row =  [cell.wafer,cell.center.x,cell.center.y,cell.center.z,cell.c,\
                        triggercell.id,triggercell.wafer,triggercell.center.x,triggercell.center.y,\
                        triggercell.center.z,triggercell.tc]
                flatlist.append(row)
    ######################################################            
                
                
                
    
    flatlist = pd.DataFrame.from_records(flatlist, columns=labels)
    if not outputFileName is None:
        flatlist.to_pickle(outputFileName)
    return flatlist

def checkishalf(temp,r):
    ishalf = []
    for index, row in temp.iterrows():
        neighbor_c = temp[np.sqrt((temp.c_x-row.c_x)**2+(temp.c_y-row.c_y)**2)<r].c_c
        if (neighbor_c.size>4):
            ishalf.append(False)
        else: 
            ishalf.append(True)
    ishalf = np.array(ishalf)
    return ishalf


def EventTcTable(df,evtid):
    cut = (df.tc_zside[evtid]==1) #& (df_.tc_layer[evtid]==3)
    
    tc_energy = df.tc_energy[evtid][cut]
    tc_theta = 2*np.arctan(np.exp(- df.tc_eta[evtid][cut]))
    tc_z = df.tc_z[evtid][cut]
    tc_r = tc_z * np.tan(tc_theta)
    tc_x = tc_r * np.cos(df.tc_phi[evtid][cut])
    tc_y = tc_r * np.sin(df.tc_phi[evtid][cut])
    tc_subdet = df.tc_subdet[evtid][cut]
    tc_layer = (-8*tc_subdet**2 + 84*tc_subdet -180)+df.tc_layer[evtid][cut] #if tc_subdet==3 else df.tc_layer[evtid]+24
    tc_wafer = df.tc_wafer[evtid][cut]
    tc_cell  = df.tc_cell[evtid][cut]
    tc_label = np.floor(tc_cell/16)
    #tc_waferocc = []
    tc = pd.DataFrame({"layer":tc_layer,"wafer":tc_wafer,"tc":tc_cell,"asic":tc_label,\
                       "r":tc_r,"x":tc_x,"y":tc_y,"z":tc_z,"e":tc_energy})
    return tc

    
def occupancy(tc,geom):
    waferoccupancy = []
    for l in np.unique(geom.layer):
        for w in np.unique(geom[geom.layer==l].wafer):#range(0,100):#np.unique(layertc.wafer):
            occ = 0
            wafergeom = geom[(geom.layer==l)&(geom.wafer==w)]
            wafertc   = tc[(tc.layer==l)&(tc.wafer==w)]
            occ       = np.count_nonzero(wafertc.wafer==w) # count fired tc in a wafer
            x,y,z     = wafergeom.x.mean(),wafergeom.y.mean(),wafergeom.z.mean()
            waferoccupancy.append([l,w,x,y,z,occ])
    waferoccupancy = np.array(waferoccupancy)
    return pd.DataFrame({"layer":waferoccupancy[:,0],"wafer":waferoccupancy[:,1],"x":waferoccupancy[:,2],
                         "y":waferoccupancy[:,3],"z":waferoccupancy[:,4],"occ":waferoccupancy[:,5]})




