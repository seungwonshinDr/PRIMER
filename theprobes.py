# -*- coding: utf-8 -*-
import numpy as np
import multiprocessing
import time

import Gibbs_multi_NUPACK as gibbs_multi
import Gibbs_single_NUPACK as gibbs_single
import Keq
import cpt_NUPACK as cpt
import nupack

import elite
import roulette
import crossover
import pointmut
import shiftmut


def probes(Targets, nEpochs, procNum, TargetsConc, ProbeConc, interval, Temp, popSize, nElite, pointmutProb, shiftmutProb, TargetCell):
    start_time = time.time()
    Probes = np.random.choice(4, (popSize, np.shape(Targets)[1])) + 1
    theprobes = np.zeros((0,np.shape(Targets)[1]))

    #Target들의 self secondary structure 에너지 계산.
    Ghp_target = np.zeros((1, np.shape(Targets)[0]))
    for i in range(np.shape(Targets)[0]):
        Seq1 = translate(Targets[i,:])
        rna_seqs = [Seq1[0]]

        G = float(nupack.mfe(rna_seqs, ordering=None, material='rna', dangles='some', T=37, multi=0, pseudo=False, sodium=1.0, magnesium=0.0, degenerate=False)[0][1])
        Ghp_target[0,i] = G

    Khpt = Keq.Keq(Ghp_target, Temp)
    Khp_target = Khpt.keq()

    #Genetic algorithm 시작.
    iter = 0
    while iter < nEpochs:
        #Probe들의 self secondary structure 에너지 계산.
        procs = []
        pipe_list = []
        Ghp_probe = np.zeros((0,1))
        for i in range(procNum):
            recv_end, send_end = multiprocessing.Pipe(False)

            Probe_part = Probes[int(i * (popSize / procNum)):int((i + 1) * (popSize / procNum)), :]

            proc = multiprocessing.Process(target=gibbs_single.Gibbs, args=(Probe_part, send_end))
            procs.append(proc)
            pipe_list.append(recv_end)
            proc.start()

        for proc in procs:
            proc.join()

        for x in pipe_list:
            result = np.array(x.recv())
            Ghp_probe = np.concatenate((Ghp_probe, result), axis=0)

        Khp = Keq.Keq(Ghp_probe, Temp)
        Khp_probe = Khp.keq()


        #Target과 Probe 사이의 결합 에너지 계산.
        procs = []
        pipe_list = []

        G_TargetProbe = np.zeros((0, np.shape(Targets)[0]))
        for i in range(procNum):
            recv_end, send_end = multiprocessing.Pipe(False)

            Probe_part = Probes[int(i * (popSize / procNum)):int((i + 1) * (popSize / procNum)), :]

            proc = multiprocessing.Process(target=gibbs_multi.Gibbs, args=(Probe_part, Targets, send_end))
            procs.append(proc)
            pipe_list.append(recv_end)
            proc.start()

        for proc in procs:
            proc.join()

        for x in pipe_list:
            result = np.array(x.recv())
            G_TargetProbe = np.concatenate((G_TargetProbe, result), axis=0)

        K = Keq.Keq(G_TargetProbe, Temp)
        Keqs = K.keq()
        #print(Gibbs_list)


        #각 세포들의 hybridization yield 계산.
        Cp_list, Ct_list, Ct_hp, Cp_hp = cpt.Conc(Keqs, Khp_target, Khp_probe, TargetsConc, ProbeConc)

        Cpts = np.zeros((np.shape(Probes)[0], np.shape(TargetsConc)[0]))
        for cell in range(np.shape(TargetsConc)[0]):
            for p in range(np.shape(Probes)[0]):
                Cpts[p, cell] = ProbeConc - Cp_list[p,cell] - Cp_hp[p,cell]


        #TargetCell의 Cpt 값의 상대 우위 계산.
        fitness = np.zeros((np.shape(Probes)[0], 1))
        for p in range(np.shape(Probes)[0]):
            TargetCpt = Cpts[p, int(TargetCell)]
            RestCpt = np.delete(Cpts[p, :], TargetCell).max()
            #print Cpts[p, :], np.delete(Cpts[p, :], TargetCell), TargetCpt, RestCpt
            fitness[p,0] = TargetCpt - RestCpt
            #print(fitness[p,0], Cpts[p,int(TargetCell)].max(), TargetCpt, RestCpt)

        #print('f', fitness)


        theprobe = Probes[np.where(fitness == fitness.max())[0], :]

        # print(theprobe[0,:])

        elites = elite.elitism(nElite, Probes, fitness)
        #print(elites)
        #print np.shape(elites), np.shape(Probes)
        leftovers = roulette.roulette(elites, Probes, fitness)

        #print np.shape(Probes),  np.shape(leftovers)
        newPopulation = crossover.cross(leftovers, popSize - np.shape(elites)[0])
        newPopulation = pointmut.pointmut(newPopulation, pointmutProb)
        newPopulation = shiftmut.shiftmut(newPopulation, shiftmutProb)

        Probes = np.concatenate((elites, newPopulation), axis=0)

        #print(Ghp_probe)

        if iter % interval == 0:
            print('Target =', str(TargetCell), '/ Iter =', iter)
            print("End Time =", time.time() - start_time, 'Fitness =', round(fitness.max() / ProbeConc * 100, 3), "%")
            for t in range(np.shape(Targets)[0]):
                Seq1, Seq2 = translate2(theprobe[0], Targets[t, :])
                rna_seqs = [Seq1[0], Seq2[0]]
                print("MFE =", str(t), nupack.mfe(rna_seqs, ordering=None, material='rna', dangles='some', T=37, multi=0, pseudo=False, sodium=1.0, magnesium=0.0, degenerate=False),
                      Cpts[np.where(fitness == fitness.max())[0], t][0])

            theprobes = np.concatenate((theprobes, [theprobe[0]]), axis = 0)
            #print(theprobes)

            #print(Cpts[np.where(fitness == fitness.max())[0], :])

        iter += 1

    return theprobes, fitness.max() / ProbeConc * 100

def translate(seq1):
    Seq1 = []

    for i in range(np.shape(seq1)[0]):
        if seq1[i] == 1:
            Seq1.append("A")
        elif seq1[i] == 2:
            Seq1.append("U")
        elif seq1[i] == 3:
            Seq1.append("G")
        elif seq1[i] == 4:
            Seq1.append("C")

        Seq1 = ["".join(Seq1)]


    return Seq1

def translate2(seq1, seq2):
    Seq1 = []
    Seq2 = []
    for i in range(np.shape(seq1)[0]):
        if seq1[i] == 1:
            Seq1.append("A")
        elif seq1[i] == 2:
            Seq1.append("U")
        elif seq1[i] == 3:
            Seq1.append("G")
        elif seq1[i] == 4:
            Seq1.append("C")

        Seq1 = ["".join(Seq1)]

        if seq2[i] == 1:
            Seq2.append("A")
        elif seq2[i] == 2:
            Seq2.append("U")
        elif seq2[i] == 3:
            Seq2.append("G")
        elif seq2[i] == 4:
            Seq2.append("C")

        Seq2 = ["".join(Seq2)]

    #print Seq1, Seq2

    return Seq1, Seq2