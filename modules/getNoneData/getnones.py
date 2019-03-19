import random
def main():
    with open("modules/getNoneData/data/genes.txt", "r") as genes:
        with open("modules/getNoneData/data/nonedata.txt", "w") as nonedata:
            counter = 0
            for line in genes.readlines():
                a = line.split("-")
                b = a[-1].split("	")
                seq = b[-1]
                if len(seq) > 80:
                    counter += 1
                    if counter % 3 == 0 and counter <= 1294*3:
                    # if counter <= 1294:
                        l = int(len(seq)/2)
                        if counter % 2 == 0:
                            nonedata.write(getComplement("".join(reversed(seq[l-40:l+41]))))
                        else:    
                            nonedata.write(seq[l-40:l+41])
                        nonedata.write("\n")

def getComplement(seq):
    comp = ""
    for nuc in seq:
        if nuc == "A":
            comp += "T"
        elif nuc == "T":
            comp += "A"
        elif nuc == "C":
            comp += "G"
        elif nuc == "G":
            comp += "C"
    return comp

if __name__ == "__main__":
    main()