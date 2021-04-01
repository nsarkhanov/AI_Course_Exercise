# Calculate the probability of drawing a heart or an ace
# probabilty of ace  or heart  in total card 
p_a=4/52 # probability of ace in total cards 
p_h=13/52 # probability of heart in total cards
 # but in hand there  2 possible so 
total_p_a=p_a*1/2 # in hand 
total_p_h=p_h*1/2
print(f"The probability of drawing a heart {total_p_h} or an ace {total_p_a}")