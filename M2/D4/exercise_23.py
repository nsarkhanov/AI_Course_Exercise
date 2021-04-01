def factorial(n):
    if n==1:
        return 1
    else:
        return n*factorial(n-1)
def diff_words(word):
    
    x=len(word)
    result =factorial(x)
    return result

diff_words("HEXAGON")