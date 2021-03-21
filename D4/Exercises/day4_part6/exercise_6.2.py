a="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def swap_letter (a,n=13):
    new_a=''
    for char in a:
        position=a.find(char)
        new_position=(n+position)%26
        new_a+=a[new_position]
    return new_a    
print(a)    
print(swap_letter(a))

