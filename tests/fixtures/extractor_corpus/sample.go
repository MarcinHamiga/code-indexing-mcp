package sample

type User struct {
    Name string
}

func NewUser() *User {
    return &User{}
}

const Version = 1
