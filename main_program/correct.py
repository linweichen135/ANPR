TEMPLATE7 = [True, True, True, False, False, False, False]
TEMPLATE6ab = [False, False, False, False, True]
TEMPLATE6c = [False, True, False, False, False, False]


def check7(pred_list) :
    pattern = [ pred_list[i].isalpha() for i in range(len(pred_list)) ]
    comprslt = [ (pattern[i] == TEMPLATE7[i]) for i in range(len(pattern)) ]
    error_list = [ i for i,x in enumerate(comprslt) if x == False ]
    return error_list
    
    
def check6c(pred_list) :
    pattern = [ pred_list[i].isalpha() for i in range(len(pred_list)) ]
    comprslt = [ (pattern[i] == TEMPLATE6c[i]) for i in range(len(pattern)) ]
    error_list = [ i for i,x in enumerate(comprslt) if x == False ]
    return error_list
    
    
def check6ab(pred_list) :
    pattern = [ pred_list[i].isalpha() for i in range(len(pred_list)) ]
    pattern = pattern[:5]
    comprslt = [ (pattern[i] == TEMPLATE6ab[i]) for i in range(len(pattern)) ]
    error_list = [ i for i,x in enumerate(comprslt) if x == False ]
    return error_list


def SwapConfusedCharacter(pred_list, error_list) :
    for i in error_list :
        if pred_list[i] == 'B' :
            pred_list[i] = '8'
        elif pred_list[i] == '8' :
            pred_list[i] = 'B'
        elif pred_list[i] == 'Q' :
            pred_list[i] = '0'
        elif pred_list[i] == '0' :
            pred_list[i] = 'Q'
        elif pred_list[i] == 'Z' :
            pred_list[i] = '2'
        elif pred_list[i] == '2' :
            pred_list[i] = 'Z'
        elif pred_list[i] == 'S' :
            pred_list[i] = '5'
        elif pred_list[i] == '5' :
            pred_list[i] = 'S'
        elif pred_list[i] == 'T' :
            pred_list[i] = '1'
        elif pred_list[i] == '1' :
            pred_list[i] = 'T'
        elif pred_list[i] == '7' :
            pred_list[i] = 'T'
    return pred_list


def correct(pred_list) :
    if len(pred_list) == 7 :
        pred_list = SwapConfusedCharacter(pred_list, check7(pred_list))
        return ''.join(pred_list) if len(check7(pred_list)) == 0 else 'FAIL'
    else :
        pred_list = SwapConfusedCharacter(pred_list, check6c(pred_list))
        if len(check6c(pred_list)) == 0 :
            return ''.join(pred_list)
        pred_list = SwapConfusedCharacter(pred_list, check6ab(pred_list))
        return ''.join(pred_list) if len(check6ab(pred_list)) == 0 else 'FAIL'
    
    
if __name__ == '__main__' :
    # print(check(['A','2','C','3','5','7','9']))
    # print(check(['A','2','C','3','5','z','9']))
    # print(check(['2','1','3','5','z','B']))
    print(correct(['8','B','B','5','2','B','1']))