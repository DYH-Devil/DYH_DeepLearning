def fizz_buzz_encode(i) :
    if i % 15 == 0 : return 3
    elif i % 5 == 0 : return 2
    elif i % 3 == 0 : return 1
    else : return 0

def fizz_nuzz_decode(i , prediction) :
    return [str(i) , 'fizz' , 'buzz' , 'fizznuzz'][prediction]

def helper(i) :
    print(fizz_nuzz_decode(i , fizz_buzz_encode(i)))


for i in range(1 , 16) :
    helper(i)