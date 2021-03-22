a="[Never gonna give you up]Never gonna let you down][Never gonna run around and desert you]"
def clean(a):
    a=a.replace("]","\n").replace("[","")

    return a
print(clean(a))