from csv import reader, writer

def main():
    infile = open("example.csv")
    csvReader = reader(infile)
   
    dic = {}

    for row in csvReader :
        dic[row[0]] = int(row[1])
    
    choice = input("조회(q), 수량추가(a), 수량빼기(m), 과일삭제(d), 끝내려면 엔터를 눌러주세요 : ")

    while choice != "":
        if choice.lower() == 'q' :
            check(dic)
        elif choice.lower() == 'a' :
            plus(dic)
        elif choice.lower() == 'm' :
            minus(dic)
        elif choice.lower() == 'd' :
            delete(dic)
        else :
            print("없는 기능입니다. 다시 입력해주세요.")
        
        choice = input("조회(q), 수량추가(a), 수량빼기(m), 과일삭제(d), 끝내려면 엔터를 눌러주세요 : ")
        
    infile.close() 

    outfile = open("example.csv", "w")
    
    for key in sorted(dic) :
        outfile.write("%s,%d\n" % (key, dic[key]))

    outfile.close()


def check(dic) :
    first = input("과일이름을 입력하세요, 끝내려면 엔터: ") 
    fruit = first.lower() 

    while first != "" :
        if fruit in dic :
            print("%s은(는) %d상자 있습니다." % (fruit, dic[fruit]))
        else :
            print("없는 과일 입니다.")
        
        first = input("과일이름을 입력하세요, 끝내려면 엔터: ")
        fruit = first.lower()
    return dic

        
def plus(dic) :
    first = input("과일이름을 입력하세요, 끝내려면 엔터: ")
    fruit = first.lower()
    
    while first!= "" :
        if fruit in dic :
            print("%s은(는) %d상자 있습니다." % (fruit, dic[fruit]))
            apple = input("추가할 수량을 입력하세요 :")
            if int(apple) > 0:
                dic[fruit] = dic[fruit] + int(apple)
                print("%s은(는) %d상자 있습니다." % (fruit, dic[fruit]))
            else :
                print("양수로 입력해주세요")
        else :
            option = input("없는 과일입니다. 추가하시려면 'y'를 눌러주세요 : ")
            if option.upper() == 'Y' :
                append = input("추가 하려고하는 과일의 수량을 입력하세요. : ")
                dic[fruit] = int(append)
                print("%s은(는) %d상자 있습니다." % (fruit, dic[fruit]))
                
        first = input("과일이름을 입력하세요, 끝내려면 엔터: ")
        fruit = first.lower()
    return dic
def minus(dic) :
    first = input("과일이름을 입력하세요, 끝내려면 엔터: ")
    fruit = first.lower()
    
    while first!= "" :
        if fruit in dic :
            print("%s은(는) %d상자 있습니다." % (fruit, dic[fruit]))
            apple = input("뺄 수량을 입력하세요 :")
            while apple != "" :
                if int(apple) > dic[fruit] :
                    print("재고수량이 더 적습니다. 다시 입력해주세요.")
                elif int(apple) > 0:
                    dic[fruit] = dic[fruit] - int(apple)
                    print("%s은(는) %d상자 있습니다." % (fruit, dic[fruit]))
                else :
                    print("양수로 입력해주세요")
                apple = input("뺄 수량을 입력하세요 :")
        else :
            print("없는 과일 입니다.")
            
        first = input("과일이름을 입력하세요, 끝내려면 엔터: ")
        fruit = first.lower()
                
    return dic           

def delete(dic):
    first = input("제거할 과일 이름을 입력하세요, 끝내려면 엔터: ")
    fruit = first.lower()
    
    while first != "" :
        if fruit in dic:
            dic.pop(fruit)
            print("%s는 삭제되었습니다." % fruit)
        first = input("제거할 과일 이름을 입력하세요, 끝내려면 엔터: ")
        fruit = first.lower()
    
    return dic
main()
