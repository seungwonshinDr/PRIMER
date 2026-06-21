# -*- coding: utf-8 -*-
import numpy as np
import theprobes as pb


np.random.seed(None)

Temp = 310
nEpochs = 1000
popSize = 20
nElite = 2
pointmutProb = 0.1
shiftmutProb = 0.7
TargetCell = 0

Targets = np.loadtxt('miRsequences')
TargetsConc = np.loadtxt('miRconc_reduced') / 1000000
ProbeConc = TargetsConc.sum() / np.shape(TargetsConc)[1]

procNum = 10
interval = 10

Repeats = 100
Percentage = 0
for i in range(Repeats):
    print('Repeats =', i, 'in', Repeats)
    dipss, percentage = pb.probes(Targets, nEpochs, procNum, TargetsConc, ProbeConc,
                  interval, Temp, popSize, nElite, pointmutProb, shiftmutProb, TargetCell)

    if percentage > Percentage:
        DIPSs = dipss
        Percentage = percentage
        print('-------------------> New Champion!! Percentage = ', round(Percentage, 4), '% <-------------------')
        np.savetxt('DIPSs_Target'+str(TargetCell)+'_Interval'+str(interval)+'_Repeats'+str(Repeats), DIPSs)

    else:
        print('-------------------> Lose.. Percentage = ', round(percentage, 4), '% / Champion = ', round(Percentage,4),'% <-------------------')


