import os
f_name=os.path.join(os.getcwd(),"text-files","blakepoems.txt")

def clean_syb(text):
    check_list=['\\','?','!',',','.','"','{','}','(',')',';','-','[',']','&',"'"]
    for char in check_list:
        text=text.replace(char,"").lower()
    return text

def final_text(text):
    new_text=text.split()
    return new_text

def  convert_file(text):
    dictionary ={}
    text=clean_syb(text) 
    words=final_text(text)    
    for word in words:
        
        if word in dictionary:
            dictionary[word]+=1
        else:
            dictionary[word]=1
      
    return sorted(dictionary.items())


with  open(f_name,"r") as text:
    n=0
    while n<10000: 
        raw_file=text.readline()
        if raw_file=="":
            break
        result=convert_file(raw_file)
        
        n+=1
        print(result) 
         
    
    
    
    