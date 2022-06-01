import argparse

parser = argparse.ArgumentParser(add_help=False,
        usage="%(prog)s [-h] -iv INPUT_VIRUS -o OUTPUT -p PHASES [-ih INPUT_HUMAN] [-p4] [-v VIRUS]",
        description="Malivhu - MAchine LearnIng for Virus classification and virus-HUman interaction prediction", 
    )
required = parser.add_argument_group("Required arguments")
required.add_argument("-iv", "--input_virus", required=True, type=str, 
    help="path to the input virus file")
required.add_argument("-o", "--output", required=True, type=str, 
    help="path to the output directory")
required.add_argument("-p", "--phases", required=True, type=int, 
    help="a value from 1 to 4. This number is the last phase to be executed. For example, if 3 is selected, Malivhu will execute phases 1, 2 and 3.")
mandatory4 = parser.add_argument_group("Required arguments if phase 4 is set to be executed")
mandatory4.add_argument("-ih", "--input_human", type=str, 
    help="path to the input human file. Only applies if phase 4 is set to be executed.")
mandatory4.add_argument("-v", "--virus", help="'cov1', 'cov2', 'mers'. Applies if the --phase4only (or -p4) flag is added.")
optional = parser.add_argument_group("Optional arguments")
optional.add_argument("-p4", "--phase4only", action="store_true", 
    help="flag for executing only phase 4. Only applies if phase 4 is set to be executed.")
other = parser.add_argument_group("Other arguments")
other.add_argument("-h", "--help", action="help", help="show this help message and exit")

args = parser.parse_args()

phase1 = True if args.phases >= 1 else False
phase2 = True if args.phases >= 2 else False
phase3 = True if args.phases >= 3 else False
phase4 = True if args.phases == 4 else False

if phase4:
    if args.input_human == None:
        parser.error('--input_human (-ih) is required when phase 4 is set to be executed.')
    if args.virus == None:
        parser.error('--virus (-v) is required when phase 4 is set to be executed.')

if args.phase4only and phase4:
    phase1 = False
    phase2 = False
    phase3 = False

import os
import subprocess
import joblib
import re
import numpy as np
import tensorflow as tf
from Bio import SeqIO

def execCmd(cmd, type):
    cmdList = cmd.split(" ")
    res = subprocess.run(cmdList, capture_output=True)
    resString = res.stdout.decode("utf-8")
    if "Error" in resString:
        print(resString.replace("Error", f"Error converting your {type} file"))
        os._exit(0)

pattern = re.compile("[\W_]+")

folder = args.output
if not os.path.exists(folder):
    os.makedirs(folder)

iFeaturePath = "./iFeature/"

virusSeqs = []
fastas = SeqIO.parse(args.input_virus, "fasta")
for rec in fastas:
    fasta = [rec.name, str(rec.seq)]
    virusSeqs.append(fasta)
    if not args.phase4only:
        with open(folder + "/VIRUS_" + pattern.sub("", rec.name) + ".fasta", "w") as f:
            f.write(">" + rec.name + "\n" + str(rec.seq))

if phase4:
    humanSeqs = []
    fastas = SeqIO.parse(args.input_human, "fasta")
    for rec in fastas:
        fasta = [rec.name, str(rec.seq)]
        humanSeqs.append(fasta)
        if not args.phase4only:
            with open(folder + "/HUMAN_" + pattern.sub("", rec.name) + ".fasta", "w") as f:
                f.write(">" + rec.name + "\n" + str(rec.seq))

if phase1:
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_virus} --out {folder}/DDEVirus.tsv --type DDE", "virus")
if phase2:
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_virus} --out {folder}/DPCVirus.tsv --type DPC", "virus")
if phase3:
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_virus} --out {folder}/CKSAAPVirus.tsv --type CKSAAP", "virus")
if phase4:
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_virus} --out {folder}/CTDDVirus.tsv --type CTDD", "virus")
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_virus} --out {folder}/PAACVirus.tsv --type PAAC", "virus")
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_human} --out {folder}/CTDDHuman.tsv --type CTDD", "human")
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_human} --out {folder}/CKSAAPHuman.tsv --type CKSAAP", "human")
    execCmd(f"python3 {iFeaturePath}/iFeature.py --file {args.input_human} --out {folder}/PAACHuman.tsv --type PAAC", "human")

if phase1:
    # Run phase 1
    X = []
    labels = []
    with open(folder + "/DDEVirus.tsv", "r") as f:
        line = f.readline()
        line = f.readline()
        while line != None and line.strip() != "":
            splits = line.split()
            labels.append(splits[0])
            splits = np.array(splits[1:]).astype(float)
            X.append(splits)
            line = f.readline()
    with open(folder + "/DPCVirus.tsv", "r") as f:
        count = 0
        line = f.readline()
        line = f.readline()
        while line != None and line.strip() != "":
            splits = line.split()[1:]
            splits = np.array(splits).astype(float)
            X[count] = np.append(X[count], splits)
            line = f.readline()
            count += 1
    X = np.array(X).astype(np.float32)

    scaler = joblib.load("scaler1.pkl")
    X = scaler.transform(X)

    model = tf.keras.models.load_model("Phase1.h5")
    X = tf.convert_to_tensor(X)
    X = tf.expand_dims(X, axis=-1)
    y_pred = model.predict(X)
    with open(folder + "/phase1.out", "w") as f:
        f.write("PROTEIN\tnon-ssRNA(-)\tssRNA(-)\n")
        for i in range(len(y_pred)):
            f.write(labels[i] + "\t")
            pred = y_pred[i]
            for score in pred:
                f.write(str(score) + "\t")
            f.write("\n")

if phase2:
    # Run phase 2
    labelsAux = []
    for i in range(len(y_pred)):
        if y_pred.argmax(axis=1)[i] == 1:
            labelsAux.append(labels[i])
    X = []
    labels = []
    if len(labelsAux) == 0:
        with open(folder + "/phase2.out", "w") as f:
            #Terminate script if no sequences passed the previous phase
            f.write("No sequences passed phase 1.")
            os._exit(0)
    with open(folder + "/CKSAAPVirus.tsv", "r") as f:
        line = f.readline()
        line = f.readline()
        while line != None and line.strip() != "":
            splits = line.split()
            if splits[0] in labelsAux:
                labels.append(splits[0])
                splits = np.array(splits[1:]).astype(float)
                X.append(splits)
            line = f.readline()
    X = np.array(X).astype(np.float32)
    scaler = joblib.load("scaler2.pkl")
    X = scaler.transform(X)
    model = tf.keras.models.load_model("Phase2.h5")
    X = tf.convert_to_tensor(X)
    X = tf.expand_dims(X, axis=-1)
    y_pred = model.predict(X)
    with open(folder + "/phase2.out", "w") as f:
        f.write("PROTEIN\tnon-Coronaviridae\tCoronaviridae\n")
        for i in range(len(y_pred)):
            f.write(labels[i] + "\t")
            pred = y_pred[i]
            for score in pred:
                f.write(str(score) + "\t")
            f.write("\n")

if phase3:
    # Run phase 3
    labelsAux = []
    for i in range(len(y_pred)):
        if y_pred.argmax(axis=1)[i] == 1:
            labelsAux.append(labels[i])
    X = []
    labels = []
    if len(labelsAux) == 0:
        with open(folder + "/phase3.out", "w") as f:
            #Terminate script if no sequences passed the previous phase
            f.write("No sequences passed phase 2.")
            os._exit(0)
    with open(folder + "/CKSAAPVirus.tsv", "r") as f:
        line = f.readline()
        line = f.readline()
        while line != None and line.strip() != "":
            splits = line.split()
            if splits[0] in labelsAux:
                labels.append(splits[0])
                splits = np.array(splits[1:]).astype(float)
                X.append(splits)
            line = f.readline()

    X = np.array(X).astype(np.float32)
    scaler = joblib.load("scaler3.pkl")
    X = scaler.transform(X)

    model = tf.keras.models.load_model("Phase3.h5")
    X = tf.convert_to_tensor(X)
    X = tf.expand_dims(X, axis=-1)
    y_pred = model.predict(X)
    with open(folder + "/phase3.out", "w") as f:
        f.write("PROTEIN\tnon-SARS/MERS\tSARS\tMERS\n")
        for i in range(len(y_pred)):
            f.write(labels[i] + "\t")
            pred = y_pred[i]
            for score in pred:
                f.write(str(score) + "\t")
            f.write("\n")

# Run phase 4
if phase4:
    X = []
    labelsAuxSARS = []
    labelsSARS1 = []
    labelsSARS2 = []
    labelsMERS = []
    if phase3:
        for i in range(len(y_pred)):
            if y_pred.argmax(axis=1)[i] == 1:
                labelsAuxSARS.append(labels[i])
        for label in labelsAuxSARS:
            virusClass = "sars1"
            modLabel = pattern.sub("", label)
            os.system("blastp -db ./db/SARS1DB -query " + folder + "/VIRUS_" + modLabel + ".fasta -out " + folder + "/VIRUS_" + modLabel + ".sars1.blast -outfmt 6 -num_threads 4")
            os.system("blastp -db ./db/SARS2DB -query " + folder + "/VIRUS_" + modLabel + ".fasta -out " + folder + "/VIRUS_" + modLabel + ".sars2.blast -outfmt 6 -num_threads 4")
            with open(folder + "/VIRUS_" + modLabel + ".sars1.blast") as f:
                greatestScore = 0
                greatestPct = 0.0
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    pct = float(splits[2])
                    score = int(splits[11])
                    if greatestScore < score:
                        greatestScore = score
                        greatestPct = pct
                    line = f.readline()
            with open(folder + "/VIRUS_" + modLabel + ".sars2.blast") as f:
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    pct = float(splits[2])
                    score = int(splits[11])
                    if greatestScore < score or (greatestScore == score and greatestPct <= pct):
                        virusClass = "sars2"
                        break
                    line = f.readline()
            if virusClass == "sars1":
                labelsSARS1.append(label)
            else:
                labelsSARS2.append(label)

        for i in range(len(y_pred)):
            if y_pred.argmax(axis=1)[i] == 2:
                labelsMERS.append(labels[i])

        if len(labelsSARS1) == 0 and len(labelsSARS2) == 0 and len(labelsMERS) == 0:
            with open(folder + "/phase3.out", "w") as f:
                #Terminate script if no sequences passed the previous phase
                f.write("No sequences passed phase 2.")
                os._exit(0)
        #SARS1
        if len(labelsSARS1) > 0:
            labelsVirus = []
            XVirus = []
            labels = []
            X = []
            with open(folder + "/CTDDVirus.tsv", "r") as f:
                line = f.readline()
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    label = splits[0]
                    if label in labelsSARS1:
                        labelsVirus.append(label)
                        splits = np.array(splits[1:]).astype(float)
                        XVirus.append(splits)
                    line = f.readline()
            for i in range(len(labelsVirus)):
                with open(folder + "/CTDDHuman.tsv", "r") as f:
                    line = f.readline()
                    line = f.readline()
                    while line != None and line.strip() != "":
                        splits = line.split()
                        labels.append(labelsVirus[i] + "___" + splits[0])
                        splits = np.array(splits[1:]).astype(float)
                        Xpair = np.append(XVirus[i], splits)
                        X.append(Xpair)
                        line = f.readline()
            
            X = np.array(X).astype(np.float32)
            model = joblib.load("CoV1.pkl")
            y_pred = model.predict_proba(X)
            with open(folder + "/phase4.out", "a") as f:
                f.write("PROTEIN\tNEGATIVE\tPOSITIVE\n")
                for i in range(len(y_pred)):
                    f.write(labels[i] + "\t")
                    f.write("SARS1 \t")
                    pred = y_pred[i]
                    for score in pred:
                        f.write(str(score) + "\t")
                    f.write("\n")

        #SARS2
        if len(labelsSARS2) > 0:
            labelsVirus = []
            XVirus = []
            labels = []
            X = []
            with open(folder + "/CKSAAPVirus.tsv", "r") as f:
                line = f.readline()
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    label = splits[0]
                    if label in labelsSARS2:
                        labelsVirus.append(label)
                        splits = np.array(splits[1:]).astype(float)
                        XVirus.append(splits)
                    line = f.readline()
            for i in range(len(labelsVirus)):
                with open(folder + "/CKSAAPHuman.tsv", "r") as f:
                    line = f.readline()
                    line = f.readline()
                    while line != None and line.strip() != "":
                        splits = line.split()
                        labels.append(labelsVirus[i] + "___" + splits[0])
                        splits = np.array(splits[1:]).astype(float)
                        Xpair = np.append(XVirus[i], splits)
                        X.append(Xpair)
                        line = f.readline()
            X = np.array(X).astype(np.float32)
            scaler = joblib.load("scalerCoV2.pkl")
            X = scaler.transform(X)
            model = tf.keras.models.load_model("CoV2.h5")
            X = tf.convert_to_tensor(X)
            X = tf.expand_dims(X, axis=-1)
            y_pred = model.predict(X)
            with open(folder + "/phase4.out", "a") as f:
                f.write("PROTEIN\tNEGATIVE\tPOSITIVE\n")
                for i in range(len(y_pred)):
                    f.write(labels[i] + "\t")
                    f.write("SARS2 \t")
                    pred = y_pred[i]
                    for score in pred:
                        f.write(str(score) + "\t")
                    f.write("\n")

        #MERS
        if len(labelsMERS) > 0:
            labelsVirus = []
            XVirus = []
            labels = []
            X = []
            with open(folder + "/PAACVirus.tsv", "r") as f:
                line = f.readline()
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    label = splits[0]
                    if label in labelsMERS:
                        labelsVirus.append(label)
                        splits = np.array(splits[1:]).astype(float)
                        XVirus.append(splits)
                    line = f.readline()
            for i in range(len(labelsVirus)):
                with open(folder + "/PAACHuman.tsv", "r") as f:
                    line = f.readline()
                    line = f.readline()
                    while line != None and line.strip() != "":
                        splits = line.split()
                        labels.append(labelsVirus[i] + "___" + splits[0])
                        splits = np.array(splits[1:]).astype(float)
                        Xpair = np.append(XVirus[i], splits)
                        X.append(Xpair)
                        line = f.readline()
            X = np.array(X).astype(np.float32)
            model = joblib.load("MERS.pkl")
            y_pred = model.predict_proba(X)
            with open(folder + "/phase4.out", "a") as f:
                f.write("PROTEIN\tNEGATIVE\tPOSITIVE\n")
                for i in range(len(y_pred)):
                    f.write(labels[i] + "\t")
                    f.write("MERS \t")
                    pred = y_pred[i]
                    for score in pred:
                        f.write(str(score) + "\t")
                    f.write("\n")

    elif args.virus == "cov1":
        labelsVirus = []
        XVirus = []
        labels = []
        X = []
        with open(folder + "/CTDDVirus.tsv", "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line.strip() != "":
                splits = line.split()
                labelsVirus.append(splits[0])
                splits = np.array(splits[1:]).astype(float)
                XVirus.append(splits)
                line = f.readline()
        for i in range(len(labelsVirus)):
            with open(folder + "/CTDDHuman.tsv", "r") as f:
                line = f.readline()
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    labels.append(labelsVirus[i] + "___" + splits[0])
                    splits = np.array(splits[1:]).astype(float)
                    Xpair = np.append(XVirus[i], splits)
                    X.append(Xpair)
                    line = f.readline()
        X = np.array(X).astype(np.float32)
        model = joblib.load("CoV1.pkl")
        y_pred = model.predict_proba(X)
        with open(folder + "/phase4.out", "a") as f:
            f.write("PROTEIN\tNEGATIVE\tPOSITIVE\n")
            for i in range(len(y_pred)):
                f.write(labels[i] + "\t")
                f.write("SARS1 \t")
                pred = y_pred[i]
                for score in pred:
                    f.write(str(score) + "\t")
                f.write("\n")

    elif args.virus == "cov2":
        labelsVirus = []
        XVirus = []
        labels = []
        X = []
        with open(folder + "/CKSAAPVirus.tsv", "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line.strip() != "":
                splits = line.split()
                labelsVirus.append(splits[0])
                splits = np.array(splits[1:]).astype(float)
                XVirus.append(splits)
                line = f.readline()
        for i in range(len(labelsVirus)):
            with open(folder + "/CKSAAPHuman.tsv", "r") as f:
                line = f.readline()
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    labels.append(labelsVirus[i] + "___" + splits[0])
                    splits = np.array(splits[1:]).astype(float)
                    Xpair = np.append(XVirus[i], splits)
                    X.append(Xpair)
                    line = f.readline()
        X = np.array(X).astype(np.float32)
        scaler = joblib.load("scalerCoV2.pkl")
        X = scaler.transform(X)
        model = tf.keras.models.load_model("CoV2.h5")
        X = tf.convert_to_tensor(X)
        X = tf.expand_dims(X, axis=-1)
        y_pred = model.predict(X)
        with open(folder + "/phase4.out", "a") as f:
            f.write("PROTEIN\tNEGATIVE\tPOSITIVE\n")
            for i in range(len(y_pred)):
                f.write(labels[i] + "\t")
                f.write("SARS2 \t")
                pred = y_pred[i]
                for score in pred:
                    f.write(str(score) + "\t")
                f.write("\n")

    elif predictVirus == "MERS":
        labelsVirus = []
        XVirus = []
        labels = []
        X = []
        with open(folder + "/PAACVirus.tsv", "r") as f:
            line = f.readline()
            line = f.readline()
            while line != None and line.strip() != "":
                splits = line.split()
                labelsVirus.append(splits[0])
                splits = np.array(splits[1:]).astype(float)
                XVirus.append(splits)
                line = f.readline()
        for i in range(len(labelsVirus)):
            with open(folder + "/PAACHuman.tsv", "r") as f:
                line = f.readline()
                line = f.readline()
                while line != None and line.strip() != "":
                    splits = line.split()
                    labels.append(labelsVirus[i] + "___" + splits[0])
                    splits = np.array(splits[1:]).astype(float)
                    Xpair = np.append(XVirus[i], splits)
                    X.append(Xpair)
                    line = f.readline()
        X = np.array(X).astype(np.float32)
        model = joblib.load("MERS.pkl")
        y_pred = model.predict_proba(X)
        with open(folder + "phase4.out", "a") as f:
            f.write("PROTEIN\tNEGATIVE\tPOSITIVE\n")
            for i in range(len(y_pred)):
                f.write(labels[i] + "\t")
                f.write("MERS \t")
                pred = y_pred[i]
                for score in pred:
                    f.write(str(score) + "\t")
                f.write("\n")