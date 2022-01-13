#!/usr/bin/env python3

from coffea import hist, util
from coffea.processor import accumulate
import numpy as np
import uproot
import os

from ttgamma.utils.plotting import RebinHist, SetRangeHist

# NOTE: your timestamps will differ!
outputMC = accumulate(
    [
        util.load("Outputs/output_MCTTGamma_run20211216_114449.coffea"),
        util.load("Outputs/output_MCSingleTop_run20211216_124610.coffea"),
        util.load("Outputs/output_MCTTbar1l_run20211216_121317.coffea"),
        util.load("Outputs/output_MCTTbar2l_run20211216_122542.coffea"),
        util.load("Outputs/output_MCWJets_run20211216_142316.coffea"),
        util.load("Outputs/output_MCZJets_run20211216_130754.coffea"),
        util.load("Outputs/output_MCOther_run20211216_165125.coffea"),
    ]
)

outputData = util.load("Outputs/output_Data_run20211216_171828.coffea")

grouping_cat = {
    "Prompt": slice(1, 2),
    "MisID": slice(2, 3),
    "NonPrompt": slice(3, 5),
}

# No QCD and TGJets?
grouping_dataset = {
    "ttgamma": [
        "TTGamma_Dilepton",
        "TTGamma_SingleLept",
        "TTGamma_Hadronic",
    ],
    "other": [
        "TTbarPowheg_Dilepton",
        "TTbarPowheg_Semilept",
        "TTbarPowheg_Hadronic",
        "W1jets",
        "W2jets",
        "W3jets",
        "W4jets",
        "DYjetsM10to50",
        "DYjetsM50",
        "ST_s_channel",
        "ST_tW_channel",
        "ST_tbarW_channel",
        "ST_tbar_channel",
        "ST_t_channel",
        "TTWtoLNu",
        "TTWtoQQ",
        "TTZtoLL",
        #"GJets_HT40To100",
        #"GJets_HT100To200",
        #"GJets_HT200To400",
        #"GJets_HT400To600",
        #"GJets_HT600ToInf",
        "ZZ",
        "WZ",
        "WW",
        #"QCD.",
        #"TGJets",

    ],
    "WG": [
        "WGamma",
    ],
    "ZG": [
        "ZGamma_01J_5f_lowMass",
    ],

}

if __name__ == "__main__":

    h = outputMC["M3"].sum("lepFlavor")
    h = h.group(
        "category", hist.Cat(r"category", r"Samples", sorting="placement"), grouping_cat
    )
    h = h.group(
        "dataset", hist.Cat(r"dataset", r"Samples", sorting="placement"), grouping_dataset
    )
    h = RebinHist(h, "M3", 10)
    h = SetRangeHist(h, "M3", 50, 550)

    hData = outputData["M3"].sum("lepFlavor")
    hData = hData.sum("dataset")
    hData = hData.sum("category")
    hData = hData.sum("systematic")
    hData = RebinHist(hData, "M3", 10)
    hData = SetRangeHist(hData, "M3", 50, 550)

    outdir = "RootFiles"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outputFile = uproot.recreate(os.path.join(outdir, "M3_Output.root"))

    outputFile["data_obs"] = hData.to_hist()

    systematics = h.axis("systematic").identifiers()
    for _category in ["MisID", "NonPrompt"]:
        for _systematic in systematics:
            histname = f"{_category}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_category}"
            outputFile[histname] = (
                h.integrate("category", _category)
                .integrate("dataset")
                .integrate("systematic", _systematic)
                .to_hist()
            )
    for _dataset in ["ttgamma", "WG", "ZG", "other"]:
        for _systematic in systematics:
            histname = f"{_dataset}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_dataset}"
            outputFile[histname] = (
                h.integrate("dataset", _dataset)
                .integrate("category", "Prompt")
                .integrate("systematic", _systematic)
                .to_hist()
            )

    outputFile.close()

    '''
    nonprompt control region
    '''

    bins = np.array([1.15, 2.5, 4.9, 9, 14.9, 20])  # 1.14 is in the cutbased medium ID.
    # regroup for the different photon categories, summing over all data sets.
    h = outputMC["photon_chIso"].sum("lepFlavor")
    h = h.group(
        "category", hist.Cat(r"category", r"Samples", sorting="placement"), grouping_cat
    )
    h = h.group(
        "dataset", hist.Cat(r"dataset", r"Samples", sorting="placement"), grouping_dataset
    )
    h = h.rebin("chIso", hist.Bin("chIso", h.axis("chIso").label, bins))

    hData = outputData["photon_chIso"].sum("lepFlavor")
    hData = hData.sum("dataset")
    hData = hData.sum("category")
    hData = hData.sum("systematic")
    hData = hData.rebin("chIso", hist.Bin("chIso", hData.axis("chIso").label, bins))

    hData.sum("chIso").values()

    outputFile = uproot.recreate(os.path.join(outdir, "Isolation_Output.root"))
    outputFile["data_obs"] = hData.to_hist()

    systematics = h.axis("systematic").identifiers()
    for _category in ["MisID", "NonPrompt"]:
        for _systematic in systematics:
            histname = f"{_category}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_category}"
            outputFile[histname] = (
                h.integrate("category", _category)
                .integrate("dataset")
                .integrate("systematic", _systematic)
                .to_hist()
            )
    for _dataset in ["ttgamma", "WG", "ZG", "other"]:
        for _systematic in systematics:
            histname = f"{_dataset}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_dataset}"
            outputFile[histname] = (
                h.integrate("dataset", _dataset)
                .integrate("category", "Prompt")
                .integrate("systematic", _systematic)
                .to_hist()
            )

    outputFile.close()

    '''
    Mis-ID control region
    '''

    h = outputMC["photon_lepton_mass_3j0t"]
    h = h.group(
        "category", hist.Cat(r"category", r"Samples", sorting="placement"), grouping_cat
    )
    h = h.group(
        "dataset", hist.Cat(r"dataset", r"Samples", sorting="placement"), grouping_dataset
    )
    h = RebinHist(h, "mass", 20)
    h = SetRangeHist(h, "mass", 40, 200)

    hData = outputData["photon_lepton_mass_3j0t"]
    hData = hData.sum("dataset")
    hData = hData.sum("category")
    hData = hData.sum("systematic")
    hData = RebinHist(hData, "mass", 20)
    hData = SetRangeHist(hData, "mass", 40, 200)


    systematics = h.axis("systematic").identifiers()

    for _lepton in ["electron", "muon"]:
        outputFile = uproot.recreate(os.path.join(outdir, f"MisID_Output_{_lepton}.root"))

        outputFile["data_obs"] = hData.integrate("lepFlavor", _lepton).to_hist()

        categories = h.axis("category").identifiers()
        systematics = h.axis("systematic").identifiers()
        for _category in ["MisID", "NonPrompt"]:
            for _systematic in systematics:
                histname = f"{_category}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_category}"
                outputFile[histname] = (
                    h.integrate("category", _category)
                    .integrate("systematic", _systematic)
                    .integrate("lepFlavor", _lepton)
                    .integrate("dataset")
                    .to_hist()
                )
        for _dataset in ["ttgamma", "WG", "ZG", "other"]:
            for _systematic in systematics:
                histname = f"{_dataset}_{_systematic}" if (not f"{_systematic}" == 'nominal') else f"{_dataset}"
                outputFile[histname] = (
                    h.integrate("dataset", _dataset)
                    .integrate("systematic", _systematic)
                    .integrate("lepFlavor", _lepton)
                    .integrate("category", "Prompt")
                    .to_hist()
                )

        outputFile.close()
