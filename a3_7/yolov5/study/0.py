b = 2
p = b**29
print(p)
r = 0

for i in range(30):
    print(i)
    r += b**i
print(r)

print(2**30-1)