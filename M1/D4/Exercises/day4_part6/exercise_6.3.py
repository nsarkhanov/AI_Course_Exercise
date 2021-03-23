text = """How much wood would a woodchuck chuck 
If a woodchuck could chuck wood?
He would chuck, he would, as much as he could,
And chuck as much as a woodchuck would
If a Mr. Smith could chuck wood\n\r\t."""

# read whole text
# create a counter
# get rid \n
# get rid of ?. special grammar
# lowercase my sentence
#"if a woodchuck could chuck wood"
# split the string by some character
#["if", "a", "woodchuck"]
#check if wood is in the list
    # if yes
       # counter = counter +1 <--> counter += 1 
    # else
        #pass
#return
def wood_counter(text):
    text = text.replace("?", "").replace(".","")
    l = text.lower().strip().split()
    counter = 0
    for word in l:
        if word == "wood":
            counter +=1
    return counter
print("total wood in text:" ,wood_counter(text))