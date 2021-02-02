import os 

a = os.path.realpath(__file__)

print(a)

print(a.split("/"))


paths = []
for i in a.split("/"):
    if i:
        if i != "hotel_cloud":
            paths.append(i)
        else:
            paths.append(i)
            break

print(paths)

print(os.path.join("/", *paths))
