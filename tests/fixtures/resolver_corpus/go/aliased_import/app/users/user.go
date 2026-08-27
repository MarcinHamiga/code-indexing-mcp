package users

type User struct {
	Name string
}

func GetByName(name string) *User {
	return &User{Name: name}
}
