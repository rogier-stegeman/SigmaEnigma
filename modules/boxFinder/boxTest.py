import boxFinder as bf
sigma_list = []
none_list = []
with open("data/sigma_data_backup.csv", "r") as data:
    for line in data:
        line = line.split(",")
        if line[82] == "Sigma70\n":
            sigma_list.append("".join(line[1:82]))
        else:
            none_list.append("".join(line[1:82]))
# print(sigma_list[len(sigma_list)-3:])
# print(none_list[len(none_list)-3:])
sigma_res = []
for seq in sigma_list:
    sigma_res.append(bf.boxFinder(seq))
r1,r2,r3,r4 = 0,0,0,0
for n in sigma_res:
    if n[0] == True:
        r1 += 1
    if n[1] == True:
        r2 += 1
    if n[2] == True:
        r3 += 1
    if n[0:3] == [False, False, False]:
        r4 += 1
print(f"""
Total sigma seqs:  {len(sigma_list)}
-10boxes found:    {r3}
-10extboxes found: {r2}
-35boxes found:    {r1}
No boxes found:    {r4}
""")    
none_res = []
for seq in none_list:
    none_res.append(bf.boxFinder(seq))
r1,r2,r3,r4 = 0,0,0,0
for n in none_res:
    if n[0] == True:
        r1 += 1
    if n[1] == True:
        r2 += 1
    if n[2] == True:
        r3 += 1
    if n[0:3] == [False, False, False]:
        r4 += 1
print(f"""
Total none seqs:  {len(none_list)}
-10boxes found:    {r3}
-10extboxes found: {r2}
-35boxes found:    {r1}
No boxes found:    {r4}
""")    