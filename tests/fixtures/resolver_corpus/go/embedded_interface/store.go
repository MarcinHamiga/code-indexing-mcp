package store

type Reader interface {
	Read(p []byte) (int, error)
}

type LogStore struct {
	Reader
	name string
}
