# Mermaid 실습
- 순서도 실습
    - 첫번재 샘플
    
```{mermaid}
classDiagram
direction LR
    class Person{
        <<abstract>>
        +name : str
        +phoneNumber : str
        +emailAddress : str
        +purchaseParkingPass()
    }

    class Address {
        +street : str
        +city : str
        +state : str
        +postalCode : str
        +country : str
        -validate() bool
        +outputAsLabel() str
    }

    class Student {
        +studentNumber : int
        +averageMark : int
        +isEligibleToEnroll(str) bool
        +getseminarsTaken() : int
    }

    class Professor {
        -salary : int
        #staffNumber : int
        -yearsOfService : int
        +numberOfClasses : int
    }

    Student "0...*" <-- "1...5" Professor : supervise
    Person "0.. 1" --> "1" Address : lives at
    Person <|-- Student
    Person <|-- Professor
```


## 여기는 끝 입니다.