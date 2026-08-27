package sample

import (
	"fmt"
	st "app/store"
	. "app/util"
)

const Version = 1

type Reader interface {
	Read(p []byte) (int, error)
}

type User struct {
	Name string
}

type Proxy struct {
	User
	next *Proxy
}

func NewUser() *User {
	return &User{Name: "n"}
}

func (u *User) Rename(name string) {
	u.Name = name
}

func (p *Proxy) Fallback() string {
	if p.next != nil {
		return p.next.Fallback()
	}
	return p.User.Name
}

func Run(r Reader) error {
	user := NewUser()
	user.Rename("x")
	total := st.Total
	fmt.Println(total)
	_ = r
	_ = user
	return nil
}
