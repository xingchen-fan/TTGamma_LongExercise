#!/usr/bin/env python3
"""
First source LCG environment to get ROOT:
source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh
"""
from ROOT import TFile, TFractionFitter, TObjArray

import pprint

_file = TFile("../RootFiles/M3_Output.root")

def getSafe(name):
    h = _file.Get(name)
    if h == None:
        print(h)
        raise KeyError(name)
    return h


systematics  = ["nominal",
                "FSRDown",
                "FSRUp",
                "ISRDown",
                "ISRUp",
                "JERDown",
                "JERUp",
                "JESDown",
                "JESUp",
                "PDFDown",
                "PDFUp",
                "Q2ScaleDown",
                "Q2ScaleUp",
                "btagWeight_heavyDown",
                "btagWeight_heavyUp",
                "btagWeight_lightDown",
                "btagWeight_lightUp",
                "eleEffWeightDown",
                "eleEffWeightUp",
                "muEffWeightDown",
                "muEffWeightUp",
                "puWeightDown",
                "puWeightUp",
]

results = {}

data = getSafe("dataObs")
    
for syst in systematics:

    mc = TObjArray(2)
    mc.Add(getSafe(f"TopPair_{syst}"))
    mc.Add(getSafe(f"NonTop_{syst}"))

    fit = TFractionFitter(data, mc,"q")
    
    status = int(fit.Fit())

    if not status==0:
        print (f"Error in fit while processing {syst} sample: exit status {status}")
    topPurity = fit.GetFitter().Result().Parameters()[0]
    topPurityErr = fit.GetFitter().Result().ParError(0)

    results[syst] = (topPurity, topPurityErr)

    del fit


pp = pprint.PrettyPrinter(indent=4)
pprint.pprint(results)
