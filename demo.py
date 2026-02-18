def catAndMouse(x, y, z):
    if(x < z):
        c1=int(z-x)
    else:
        c1=int(x-z)
    print("c11: ",c1)
    if(y < z):
        c2=int(z-y)
    else:
        c2=int(y-z)
    print("c2: ",c2)
    if(c2>c1):
        return "CatA"
    elif(c2<c1):
        return "CatB"
    else:
        return "MouseC"
xyz = input("Enter Seqence:").split()

x = int(xyz[0])

y = int(xyz[1])

z = int(xyz[2])

result = catAndMouse(x, y, z)
print(result + '\n')